# Durable Functions Agents Sample

Trigger Microsoft Agent Framework agents from an Azure Durable Function. An HTTP
request launches a durable orchestration that coordinates multiple cooperating
agents, persists their conversation inside a Durable Entity, and demonstrates how
an agent can expose another agent as a tool.

## Scenario

1. **HTTP trigger** starts the orchestration with a release-readiness topic.
2. The orchestrator initializes a Durable Entity (`AgentConversationState`).
3. An activity function creates a coordinator agent that delegates research to a
   specialist agent via a custom tool. The research agent can in turn ask a critic
   agent to review findings. Each tool invocation captures structured state for the
   entity.
4. Results (messages, tasks, tool calls, summary) are persisted through Durable
   Entity operations, providing a resilient audit trail for the session.

## Prerequisites

- Python 3.11+
- Azure Functions Core Tools v4
- Azurite (or a real Azure Storage account) for `AzureWebJobsStorage`
- Azure OpenAI access and environment variables for `AzureOpenAIChatClient`
  (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, and `AZURE_OPENAI_CHAT_DEPLOYMENT`)
- Install repository packages (from repo root):

  ```powershell
  uv pip install -e python/packages/core
  ```

## Quick start

1. Install dependencies and activate the recommended virtual environment for
   Functions (the workspace task `pip install (functions)` sets this up).
2. Configure `local.settings.json`:
   - `AzureWebJobsStorage`: storage connection string or `UseDevelopmentStorage=true`
   - `AZURE_OPENAI_*` environment variables (add them under `Values`)
3. From this folder run the Functions host:

   ```powershell
   func start
   ```

4. POST a request to start a conversation:

   ```powershell
   Invoke-RestMethod `
     -Uri "http://localhost:7071/api/agents/durable/start" `
     -Method POST `
     -ContentType "application/json" `
     -Body (@{
         topic = "Assess launch readiness for Contoso Forecasting";
         metadata = @{ requester = "release-manager" }
       } | ConvertTo-Json)
   ```

   The response includes status URLs for checking orchestration progress.

5. Inspect Durable Entity state by following the `statusQueryGetUri` until
   `runtimeStatus` reaches `Completed`. The JSON output includes all turns, tasks,
   tool invocations, and the final summary.

## Files

- `function_app.py` – HTTP trigger, orchestrator, activity, and Durable Entity. Logs write to `function_app.log` in this folder.
- `requirements.txt` – runtime dependencies for this sample.
- `local.settings.json` – update with storage and OpenAI credentials for local use.

## Notes

- By default the coordinator runs two planned turns plus a closing summary. Tune
  the `maxTurns` request property if you want to iterate more often.
- The sample uses Azure OpenAI. Update `function_app.py` if you prefer a different
  chat client.
- Durable Entities persist state in the storage account configured by
  `AzureWebJobsStorage`. Clear the entity by calling the `reset` operation via the
  Durable Entities HTTP API when needed.
