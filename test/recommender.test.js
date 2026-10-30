import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import * as tf from '@tensorflow/tfjs';
import {
    createTrainingData,
    createVocabulary,
    recommendTestCases,
    trainModel
} from '../src/service/RecommendationModel.js';

const loadJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));

test('trains the TensorFlow recommender and returns QA checklist fields', async () => {
    await tf.ready();
    const [testCases, history] = await Promise.all([
        loadJson('../data/test-cases.json'),
        loadJson('../data/change-history.json')
    ]);
    const vocabulary = createVocabulary(testCases, history);
    const trainingData = createTrainingData(tf, testCases, history, vocabulary);
    const model = await trainModel(tf, trainingData);

    try {
        const recommendations = recommendTestCases(tf, model, {
            branch: 'feature/inventory-selection',
            title: 'Update inventory item selection',
            description: 'Keep inventory selection in sync.',
            commitMessages: ['Fix selected inventory state']
        }, testCases, vocabulary);

        assert.equal(recommendations.length, testCases.length);
        assert.ok(recommendations[0].id);
        assert.ok(recommendations[0].functionality);
        assert.ok(recommendations[0].reason);
        assert.ok(recommendations[0].priority);
        assert.ok(recommendations[0].expectedResult);
        assert.ok(recommendations[0].gherkin.feature);
        assert.ok(Number.isFinite(recommendations[0].score));
        assert.ok(recommendations.every((item, index, list) =>
            index === 0 || list[index - 1].score >= item.score
        ));
    } finally {
        model.dispose();
    }
});

test('rejects a history without both selected and unselected cases', async () => {
    const testCases = await loadJson('../data/test-cases.json');
    const history = await loadJson('../data/change-history.json');
    const vocabulary = createVocabulary(testCases, history);

    assert.throws(
        () => createTrainingData(tf, testCases, [{
            ...history[0],
            selectedTestCaseIds: testCases.map(testCase => testCase.id)
        }], vocabulary),
        /negative examples/
    );
});

test('rejects history entries that reference unknown test case IDs', async () => {
    const testCases = await loadJson('../data/test-cases.json');
    const history = await loadJson('../data/change-history.json');
    const vocabulary = createVocabulary(testCases, history);

    assert.throws(
        () => createTrainingData(tf, testCases, [{
            ...history[0],
            selectedTestCaseIds: ['UNKNOWN-999']
        }], vocabulary),
        /unknown test case IDs: UNKNOWN-999/
    );
});

test('PR runner fetches commit context and publishes a checklist comment', async () => {
    let publishedComment;
    const apiServer = createServer(async (request, response) => {
        if (request.method === 'GET' && request.url.startsWith('/repos/demo/unity-game/pulls/42/commits')) {
            response.writeHead(200, { 'Content-Type': 'application/json' });
            response.end(JSON.stringify([{
                commit: { message: 'Fix inventory selection state' }
            }]));
            return;
        }
        if (request.method === 'GET' && request.url.startsWith('/repos/demo/unity-game/issues/42/comments')) {
            response.writeHead(200, { 'Content-Type': 'application/json' });
            response.end('[]');
            return;
        }
        if (request.method === 'POST' && request.url === '/repos/demo/unity-game/issues/42/comments') {
            let body = '';
            for await (const chunk of request) body += chunk;
            publishedComment = JSON.parse(body).body;
            response.writeHead(201, { 'Content-Type': 'application/json' });
            response.end(JSON.stringify({ id: 100 }));
            return;
        }
        response.writeHead(404);
        response.end();
    });

    await new Promise(resolve => apiServer.listen(0, '127.0.0.1', resolve));
    const tempDirectory = await mkdtemp(join(tmpdir(), 'qa-recommender-test-'));
    const eventPath = join(tempDirectory, 'event.json');
    const projectPath = fileURLToPath(new URL('../', import.meta.url));
    const scriptPath = join(projectPath, 'scripts', 'recommend-pr.js');
    await writeFile(eventPath, JSON.stringify({
        repository: { owner: { login: 'demo' }, name: 'unity-game' },
        pull_request: {
            number: 42,
            head: { ref: 'feature/inventory-selection' },
            title: 'Update inventory selection',
            body: 'Keep inventory item selection synchronized.'
        }
    }));

    try {
        const port = apiServer.address().port;
        await new Promise((resolve, reject) => {
            const child = spawn(process.execPath, [scriptPath], {
                cwd: projectPath,
                env: {
                    ...process.env,
                    GITHUB_EVENT_PATH: eventPath,
                    GITHUB_TOKEN: 'test-token',
                    GITHUB_API_URL: `http://127.0.0.1:${port}`
                },
                stdio: ['ignore', 'pipe', 'pipe']
            });
            let output = '';
            child.stdout.on('data', chunk => { output += chunk; });
            child.stderr.on('data', chunk => { output += chunk; });
            child.on('error', reject);
            child.on('close', code => {
                if (code === 0) resolve(output);
                else reject(new Error(`PR runner exited with code ${code}: ${output}`));
            });
        });

        assert.match(publishedComment, /Checklist recomendado para QA/);
        assert.match(publishedComment, /INV-00[12]/);
        assert.match(publishedComment, /Feature: Inventory/);
        assert.match(publishedComment, /Resultado esperado:/);
    } finally {
        await new Promise(resolve => apiServer.close(resolve));
        await rm(tempDirectory, { recursive: true, force: true });
    }
});
