# Tentative Design: Durable Task Provider Redesign

## Objective

Design the `agent-framework-durabletask` package to provide a safe, low-overhead, and unified experience for using agents with `durabletask`, removing the need for `AgentWorker` and `AgentClient` wrappers.

## Summary of Design Patterns Explored

During the design process, we evaluated several patterns to achieve "extensions" in Python:

1.  **Monkey Patching:** Dynamically adding methods to `TaskHubGrpcWorker`.
    *   *Verdict:* Rejected due to safety concerns and "magic" behavior.
2.  **Mixins:** Creating subclasses like `AgentWorker(TaskHubGrpcWorker, Mixin)`.
    *   *Verdict:* Rejected due to boilerplate (forcing users to subclass).
3.  **Instance Helper:** `AgentDurable(worker).register(...)`.
    *   *Verdict:* Good, but slightly more overhead than static methods.
4.  **Static Utility:** `AgentDurable.register(worker, ...)`
    *   *Verdict:* Selected as the foundation for its simplicity and explicit nature.
5.  **Agent Wrapper:** Returning a `DurableAgent` object from registration.
    *   *Verdict:* Selected to provide a safe "handle" for the registered agent.

## Final Proposed Design: The Hybrid Pattern

We will use a **Static Utility Class (`AgentDurable`)** that serves as the entry point, combined with a **`DurableAgent` Wrapper** returned upon registration to facilitate safe execution.

### 1. `_extensions.py`

```python
from typing import Callable, Optional, Any
import logging
import asyncio
from durabletask.worker import TaskHubGrpcWorker
from durabletask.client import TaskHubGrpcClient
from ._entities import AgentProtocol, create_agent_entity
from ._models import AgentResponse

logger = logging.getLogger(__name__)

class AgentDurable:
    """Static utility class for Agent Framework durability operations."""

    @staticmethod
    def register(worker: TaskHubGrpcWorker, agent: AgentProtocol) -> 'DurableAgent':
        """Registers an agent instance and returns a durable wrapper.
        
        Args:
            worker: The TaskHubGrpcWorker instance.
            agent: The agent instance to register.
            
        Returns:
            A DurableAgent wrapper that can be used to run the agent durably.
        """
        entity_class = create_agent_entity(agent)
        worker.add_entity(entity_class)
        return DurableAgent(agent=agent)

    @staticmethod
    def register_factory(worker: TaskHubGrpcWorker, factory: Callable[[], AgentProtocol]) -> None:
        """Registers an agent factory function.
        
        This pattern is safer as it prevents the user from holding a reference 
        to the raw agent instance, preventing accidental local execution.
        """
        # We need to instantiate once to get the name, or require name as an argument.
        # For simplicity in this example, assuming we instantiate to get name/class.
        temp_agent = factory()
        entity_class = create_agent_entity(temp_agent) # Logic needs to support factory
        worker.add_entity(entity_class)

    @staticmethod
    def run(
        client: TaskHubGrpcClient, 
        agent_name: str, 
        message: str, 
        thread_id: str | None = None,
        # ... other params
    ) -> AgentResponse:
        """Runs an agent by name using the provided Client.
        
        Useful when the agent instance is not available (e.g., remote client).
        """
        # Logic to run agent (ported from old AgentClient)
        # ...

class DurableAgent:
    """Wrapper that guides users toward durable execution."""
    
    def __init__(self, agent: AgentProtocol):
        self._agent = agent

    @property
    def name(self) -> str:
        return self._agent.name

    def run(
        self, 
        message: str, 
        client: Optional[TaskHubGrpcClient] = None, 
        **kwargs
    ) -> Any:
        """Executes the agent.
        
        If a 'client' is provided, the agent is executed durably via the Durable Task framework.
        If 'client' is None, the agent is executed locally (in-process) and a warning is logged.
        
        Args:
            message: The input message.
            client: The TaskHubGrpcClient to use for durable execution.
        """
        if client:
            # Durable Execution
            return AgentDurable.run(client, self.name, message, **kwargs)
        else:
            # Local Execution (Fallback)
            logger.warning(
                f"Agent '{self.name}' is running locally! State will NOT be persisted. "
                "Pass a 'client' to run durably."
            )
            # Assuming agent.run is async, we might need to handle this appropriately
            # depending on where this is called from (async vs sync context).
            # For this design doc, we assume we can call it.
            return asyncio.run(self._agent.run(message=message, **kwargs))
```

### 2. Usage Experience

**Scenario A: Standard Registration (Worker Side)**
```python
# 1. Create native worker
worker = TaskHubGrpcWorker(...)

# 2. Register agent (returns wrapper)
durable_agent = AgentDurable.register(worker, my_agent)

# 3. Start worker
worker.start()

# 4. Run Durably (Recommended)
durable_agent.run("Hello", client=my_client) 

# 5. Run Locally (Warning logged)
durable_agent.run("Hello") 
```

**Scenario B: Remote Execution (Client Side)**
```python
client = TaskHubGrpcClient(...)

# Run by name
AgentDurable.run(client, "my_agent", "Hello")
```

## Comparison: Wrapper (Mixin) vs. Current Design

| Feature | Mixin / Wrapper Design | New Hybrid Design |
| :--- | :--- | :--- |
| **Core Abstraction** | `AgentWorker` / `AgentClient` Wrappers | `AgentDurable` Static Utility |
| **Integration** | Wraps the native objects. | Uses native objects directly. |
| **Boilerplate** | High (Must instantiate wrappers). | Low (Static method calls). |
| **Safety** | Low (User holds raw agent). | Medium/High (Wrapper guides usage). |
| **Flexibility** | Rigid (Must use wrappers). | Flexible (Can use static or wrapper). |
| **API Style** | `worker.add_agent(...)` | `AgentDurable.register(worker, ...)` |

**Benefits:**
1.  **Zero Overhead:** Users don't need to change how they instantiate their `durabletask` workers or clients. They just "sprinkle in" the agent capabilities.
2.  **Explicit Durability:** The `DurableAgent` wrapper and `AgentDurable` class make it clear when you are crossing the boundary into the durable framework.
3.  **Safety Net:** The `DurableAgent.run` method actively warns users if they forget to provide a client, preventing accidental local execution while still allowing it for testing if intended.
4.  **Future Proof:** It's easier to add new static utility methods than to keep updating wrapper classes to match the underlying SDK's changes.

## Open Question / Gotchas:
1. The customers can still execute their agents directly in both the worker and the client. Though we have made some efforts to inform them for it (like the warning log), it might not always be possible to avoid it. 
2. Currently we need the `TaskHubGrpcClient` object to execute the agent durably (i.e. `AgentDurable.run(client, "my_agent", "Hello")`). If the customer wants to call the Agent from within the Worker, they would need a client object which might be anti-pattern. Is there a way to trigger an entity from the worker without needing a client? 