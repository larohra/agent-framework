# Thread Management Examples

This folder contains examples demonstrating different ways to manage conversation threads and chat message stores with the Agent Framework.

## Examples

| File | Description |
|------|-------------|
| [`custom_chat_message_store_thread.py`](custom_chat_message_store_thread.py) | Demonstrates how to implement a custom `ChatMessageStore` for persisting conversation history. Shows how to create a custom store with serialization/deserialization capabilities and integrate it with agents for thread management across multiple sessions. |
| [`durable_task_orchestrated_thread.py`](durable_task_orchestrated_thread.py) | Orchestrates agent turns through Azure Durable Task Scheduler, enabling reliable, scalable execution where each user turn is processed by a Durable activity and checkpointed in the scheduler. Includes worker/client bootstrap and continue-as-new orchestration pattern. |
| [`durable_task_chat_message_store.py`](durable_task_chat_message_store.py) | Implements a `ChatMessageStore` backed by a Durable Task orchestration that persists chat history via scheduler checkpoints. Demonstrates raising orchestration events to append messages and retrieving history from scheduler state. |
| [`durable_task_activity_chat_message_store.py`](durable_task_activity_chat_message_store.py) | Extends the Durable chat store to route writes through a Durable Task activity so the orchestrator can capture enriched state and activity metadata in custom status. Illustrates orchestrator `call_activity` usage and diagnostics-friendly persistence. |
| [`durable_task_multi_agent_thread.py`](durable_task_multi_agent_thread.py) | Demonstrates multiple agents with distinct toolsets collaborating on a shared Durable Task-backed thread. Highlights tool invocation, shared orchestration state, and inspection of activity metadata across agent roles. |
| [`suspend_resume_thread.py`](suspend_resume_thread.py) | Shows how to suspend and resume conversation threads, allowing you to save the state of a conversation and continue it later. This is useful for long-running conversations or when you need to persist conversation state across application restarts. |
| [`redis_chat_message_store_thread.py`](redis_chat_message_store_thread.py) | Comprehensive examples of using the Redis-backed `RedisChatMessageStore` for persistent conversation storage. Covers basic usage, user session management, conversation persistence across app restarts, thread serialization, and automatic message trimming. Requires Redis server and demonstrates production-ready patterns for scalable chat applications. |
