# QA Test Case Recommender

Aplicação de demonstração e workflow GitHub para recomendar casos de teste de QA a partir do contexto de Pull Requests de jogos Unity. A implementação preserva o fluxo do exemplo original: dados JSON, serviço, controlador, worker TensorFlow.js, treino supervisionado e ranking por score.

## Mapeamento do exemplo original

| Exemplo de e-commerce | Recomendador de QA |
| --- | --- |
| Produto e características | Caso de teste, funcionalidade, componente, prioridade, tipo e tags |
| Perfil do usuário | Contexto de alteração: branch, título, descrição e commits |
| Compras anteriores | Seleções de casos em `data/change-history.json` |
| Produto recomendado | Caso ordenado pelo score da rede neural |
| Lista de produtos | Checklist com justificativa, Gherkin, resultado esperado e acompanhamento |

O modelo continua sendo uma rede densa TensorFlow.js com camadas `128 → 64 → 32 → 1`, saída sigmoid e `binaryCrossentropy`. O serviço compartilhado `src/service/RecommendationModel.js` é usado tanto pelo worker da demonstração quanto pelo processo Node do workflow, evitando manter duas implementações do algoritmo.

## Estrutura

- `data/test-cases.json`: catálogo de casos de teste. Cada caso contém os campos de QA e um cenário Gherkin revisável.
- `data/change-history.json`: exemplos rotulados de contexto de alteração e IDs dos casos selecionados pela QA.
- `src/service/RecommendationModel.js`: codificação de atributos/texto, preparação dos pares, treino e ranking TensorFlow.
- `src/workers/modelTrainingWorker.js`: execução do treino/inferência sem bloquear a interface.
- `src/controller/QAController.js` e `src/view/ChecklistView.js`: fluxo MVC da demonstração local.
- `scripts/recommend-pr.js`: carrega os metadados e commits do PR, treina o modelo e cria/atualiza o comentário.
- `.github/workflows/qa-test-recommender.yml`: integração acionada em PR aberto, atualizado ou reaberto.

## Executar localmente

Requer Node.js 20 ou superior.

```sh
npm ci
npm test
npm start
```

Abra `http://localhost:3000`. A interface treina o modelo a partir do histórico de demonstração e aceita os mesmos campos que o workflow lê do GitHub.

## Dataset e qualidade das recomendações

O modelo precisa de exemplos supervisionados: cada entrada no histórico deve associar o contexto de uma alteração a casos de teste que a QA selecionou. O dataset incluído é **ilustrativo**, não representa o histórico do seu jogo. Antes de usar o score como sinal operacional:

1. Substitua os casos de demonstração por casos existentes no projeto, preservando IDs estáveis e Gherkin derivado desses casos.
2. Substitua/expanda o histórico com PRs reais e testes selecionados ou executados pela QA.
3. Atualize `selectedTestCaseIds` com IDs presentes em `test-cases.json`; mantenha casos não selecionados para que o treino tenha exemplos negativos.
4. Revise as sugestões e mantenha a decisão final de cobertura com a QA. O score é uma ordenação, não uma probabilidade calibrada nem substitui critérios ISTQB.

Os cenários não são gerados por IA: o comentário usa os campos Gherkin cadastrados em cada caso, evitando inventar funcionalidades.

## Configuração do GitHub Actions

O workflow vem pronto em `.github/workflows/qa-test-recommender.yml`:

- Eventos: PR `opened`, `synchronize` e `reopened`.
- Permissões mínimas: `contents: read`, `pull-requests: read` e `issues: write`.
- Usa o `GITHUB_TOKEN` automático; não é necessário criar PAT ou adicionar secret.
- Busca branch, título, descrição e mensagens de commits; comentários subsequentes atualizam o comentário do bot existente.
- Usa `pull_request_target` para permitir comentário também em PRs de forks. Por segurança, faz checkout **somente da branch base** e não executa código, scripts ou dependências fornecidos pelo PR.
- Em **Settings → Actions → General → Workflow permissions**, confirme que o workflow pode receber as permissões definidas no YAML. Se a política da organização restringir permissões, permita leitura de Pull Requests e escrita em Issues (comentários de PR usam a API de comentários de issues).

O arquivo do workflow e os scripts precisam estar na branch base para que a automação esteja disponível ao abrir um PR. Após mesclar, abrir ou atualizar um PR dispara a recomendação.

## Limites e próximos passos

O pipeline treina o modelo durante cada execução com o dataset versionado. Para a primeira implantação, isso evita serviço, banco de dados ou infraestrutura adicionais; conforme o histórico crescer, considere persistir exemplos e treinar em um job separado. Também é necessário definir como registrar o resultado da QA no histórico, pois o comentário atual oferece checklist manual e não coleta automaticamente as respostas.


 Re-teste com permissoes ativas