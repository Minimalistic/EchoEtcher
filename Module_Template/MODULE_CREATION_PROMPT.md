# AI Programmer Prompt: Create a New j-gents Module

**Instructions to Developer**: Copy and paste the text below into your AI coding assistant (e.g., Cursor, GitHub Copilot, ChatGPT) to instruct it how to build a new module for the j-gents platform. Fill in the `[SPECIFIC_REQUIREMENTS]` section with your unique needs.

---

**Role**: You are an expert Python developer working on the "j-gents" platform. j-gents is a modular, local-first AI agent platform built with Python, FastAPI, SQLite, and Ollama.

**Objective**: Create a new, self-contained module for the platform.

**Context**: You have been provided with `platform_context.xml` (or similar context file) which contains the core source code of the j-gents platform. Use this to understand the base classes (`BaseAgent`, `BaseModule`), database session management, and available utilities.

## Design Ethos & Philosophy
1.  **Local-First & Private**: All data should remain local. Avoid external API calls unless the module's primary purpose is fetching data (e.g., RSS, Price Monitoring).
2.  **Modular & Self-Contained**: Your module should be a folder in `modules/` that contains *everything* it needs (agent, router, UI, models). It should not pollute the core code.
3.  **Aesthetics Matter**: "j-gents" aims for a premium, modern user experience. UI components should look polished, using the platform's glassmorphism style, smooth transitions, and consistent color palette.
4.  **Simplicity**: Prefer simple, readable code. Avoid over-engineering. Use the platform's existing tools (memory, scheduler) before building your own.
5.  **Agentic Workflow**: Agents must be "task-aware". They should not just run on a loop but also be able to accept tasks from the Orchestrator.

## Platform Architecture Overview

1.  **Modules**: Live in `modules/`. Each is a directory (e.g., `modules/rss_aggregator/`).
2.  **Core Components** (Do not modify these, but import from them):
    *   `core.module_standard.BaseModule`: The entry point class.
    *   `core.agent_base.BaseAgent`: The base class for background workers.
    *   `core.module_standard.UIComponent`: Defines widgets for the dashboard.
    *   `core.database.SessionLocal`: For database access (SQLAlchemy async).

## Task: Create Module `[MODULE_NAME]`

Please create the following file structure in `modules/[module_snake_case_name]/`:

### 1. `agent.py`
Create a class inheriting from `BaseAgent`.
*   **Define `capabilities`**: In `__init__`, list what this agent can do using **JSON Schemas** for the input payloads. This enables active UI form generation and precise Orchestrator planning:
    ```python
    self.capabilities = [
        {
            "name": "fetch_data", 
            "description": "Fetches data from...", 
            "schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["url"]
            }
        }
    ]
    ```
*   **Implement `run()`**: This is the background loop (if needed) for autonomous monitoring.
*   **Implement `process_task(self, payload)`**: REQUIRED. Logic to handle a direct request from the Orchestrator.
    ```python
    async def process_task(self, payload):
        # execute logic based on payload
        return {"data": "result"}
    ```
*   **Implement `schedule()`**: Define background interval (e.g., `self.scheduler.add_job(self.run, 'interval', minutes=60)`). Task polling is handled automatically if `polling_enabled` is True (see Polling Configuration below).
*   **Logging**: Use `await self.log("message", level="INFO")` where level can be "INFO", "WARNING", "ERROR", or "DEBUG".
*   **Memory**: Use `await self.save_memory("key", value)` and `await self.get_memory("key", default=None)` for simple persistence.
*   **LLM Access**: Use `await self.llm.generate_text(prompt, model=None, system=None)` or `await self.llm.generate_json(prompt, model=None, system=None)` for AI features. Use `await self.llm.get_embedding(text, model=None)` for embeddings.
*   **Polling Configuration** (Optional): Enable task polling by setting `self.polling_enabled = True` and `self.polling_interval = 30` (seconds) in `__init__`. The platform automatically calls `load_config()` to load saved settings from memory. Use `await self.update_polling_config(enabled, seconds)` to dynamically update polling.
*   **Cleanup** (Optional): Override `async def cleanup(self):` to release resources (stop watchers, close connections, etc.). Called during factory reset and module shutdown.

### 2. `router.py` (Optional)
If the module needs API endpoints:
*   Create a `APIRouter` instance.
*   Define endpoints (GET/POST).

### 3. `module.py`
Create a class inheriting from `BaseModule`.
*   **`__init__`**:
    *   Call `super().__init__("ModuleName")`.
    *   Initialize the agent: `self.agent = MyAgent("agent_name")`.
    *   Register components:
        *   `self.register_agent(self.agent)`
        *   `self.register_router(api_router)` (if applicable)
        *   `self.register_ui(UIComponent(..., agent_name="agent_name"))`

### 4. `models.py` (Optional)
If you need custom SQL tables (more than just the simple Key-Value memory):
*   Import `Base` from `core.database`.
*   Define your classes: `class MyModel(Base): ...`
*   **Important**: In `module.py`, inside `on_startup`, you must ensure these tables are created:
    ```python
    from core.database import engine, Base
    from .models import MyTable # Import to register metadata

    async def on_startup(self, app_state):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    ```

### 5. `__init__.py`
*   Expose the module class: `from .module import MyModule`.
*   Example:
    ```python
    from .module import MyModule
    
    __all__ = ["MyModule"]
    ```

### 6. `requirements.txt` (Optional)
If your module has external dependencies:
*   List one package per line (e.g., `feedparser>=1.6.0`)
*   The platform will check for these dependencies at startup and warn if missing
*   Users should install them manually: `pip install -r modules/your_module/requirements.txt`

### 7. Inter-Agent Communication (Event Bus)
Modules should stay decoupled. Use the global `EventBus` to share state changes or trigger actions in other modules.
*   **Subscribe**: `self.event_bus.subscribe("event_name", self.callback_method)` (callback can be sync or async)
*   **Publish**: `await self.event_bus.publish("event_name", {"data": "value"})` (EventBus is a class, accessed via `self.event_bus`)
    ```python
    # Example: In your agent's run() or process_task()
    await self.event_bus.publish("file_processed", {"path": "/tmp/test.txt", "sender": self.name})
    ```
*   **Note**: Events are automatically persisted to the database as `SystemEvent` records.

## UI & Styling Guidelines (Crucial)
The platform uses a dark, glassmorphism-themed design. Your `UIComponent` `html_content` should adhere to these standards:

*   **CSS Variables**: Use these for consistency (do not hardcode hex values where possible):
    *   `--bg-color`, `--text-color` (Main background/text)
    *   `--primary-color`, `--accent-color` (Gradients/Highlights)
    *   `--glass-border` (Subtle 1px border)
*   **Layout**:
    *   The platform wraps your widget in a `.card` container automatically.
    *   Use flexbox or grid for internal layout.
*   **Module UI States**: Each module widget has 3 possible states, automatically managed by the platform:
    *   **Collapsed**: Only the header is visible (minimal vertical space)
    *   **Expanded**: Normal view with full content visible (default)
    *   **Maximized**: Fullscreen overlay mode for focused work (press ESC to exit)
    *   State is persisted to `localStorage` per module
*   **Components**:
    *   **Buttons**: Add class `btn` for the standard gradient button.
    *   **Status Indicators**: Use `span` with class `status-badge`. Add `status-running` (green) or `status-stopped` (red).
*   **Agent Integration**: 
    *   If you set `agent_name` in your `UIComponent`, the platform will **automatically**:
        *   Display a status badge (running/standby) in the widget header.
        *   Render a "Tools" section in the footer with tags that trigger dynamic action forms based on your schemas.
*   **Aesthetics**:
    *   Ensure text contrasts well (it's a dark theme).
    *   Avoid white backgrounds; use transparent or dark slate colors.

## Best Practices
1.  **Async/Await**: The core is built on asyncio. Ensure your `run()` loop and any I/O are non-blocking.
2.  **Error Handling**: Your module should be robust. Catch exceptions in `run()` to prevent crashing the main loop. Log errors using `await self.log("error message", level="ERROR")`.
3.  **Type Hinting**: Use Python type hints heavily for better AI understanding and code safety.
4.  **Dependencies**: If you need external packages (e.g., `feedparser`), list them in `requirements.txt` within your module directory.
5.  **Resource Cleanup**: Implement `cleanup()` method if your agent manages resources (file watchers, connections, etc.) to ensure proper shutdown.
6.  **Task Validation**: The platform automatically validates task payloads against your capability schemas using JSON Schema. Invalid payloads will be rejected before reaching `process_task()`.

## specific Module Requirements
[PASTE YOUR SPECIFIC REQUIREMENTS HERE]
*   **Goal**: [Describe what this module should do]
*   **Inputs**: [What data does it need?]
*   **Outputs**: [What does it produce/display?]
*   **Frequency**: [How often should it run?]
