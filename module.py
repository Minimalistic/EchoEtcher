from core.module_standard import BaseModule, UIComponent
from .agent import EchoEtcherAgent
from .router import create_router

class EchoEtcherModule(BaseModule):
    def __init__(self):
        super().__init__("EchoEtcher")
        self.agent = EchoEtcherAgent("echo_etcher")
        
        # Register components
        self.register_agent(self.agent)
        self.register_router(create_router(self.agent))
        
        # Register UI Component
        self.register_ui(UIComponent(
            id="echo_etcher_widget",
            title="Echo Etcher",
            agent_name="echo_etcher",
            html_content="""
<div class="echo-etcher-container" style="display: flex; flex-direction: column; gap: 0.5rem; height: 100%;">
    <div class="console-output" id="echo-console" 
         style="flex-grow: 1; min-height: 200px; background: #000; border-radius: 4px; padding: 0.5rem; 
                font-family: monospace; font-size: 0.75rem; color: #34d399; overflow-y: auto; border: 1px solid rgba(255,255,255,0.05);">
        <div>> Echo Etcher console ready...</div>
    </div>
</div>
""",
            script_content="""
async function pollEchoLogs() {
    try {
        const response = await fetch('/echo_etcher/logs');
        const logs = await response.json();
        const consoleDiv = document.getElementById('echo-console');
        if (!consoleDiv) return;
        
        if (logs && logs.length > 0) {
            consoleDiv.innerHTML = ''; 
            logs.forEach(msg => {
                const line = document.createElement('div');
                line.textContent = `> ${msg}`;
                consoleDiv.appendChild(line);
            });
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }
    } catch (e) {
        // ignore errors on poll
    }
}

// Init
setInterval(pollEchoLogs, 3000);
pollEchoLogs();
"""
        ))

    async def on_startup(self, app_state):
        # Could ensure tables exist here if we added custom models
        pass
    
    async def on_shutdown(self):
        """
        Clean up resources on module shutdown.
        """
        if hasattr(self.agent, 'cleanup'):
            await self.agent.cleanup()