from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

def create_router(agent):
    router = APIRouter(prefix="/echo_etcher", tags=["echo_etcher"])

    class ConfigRequest(BaseModel):
        watch_path: str
        dest_path: str

    @router.get("/config")
    async def get_config():
        return {
            "watch_path": await agent.get_memory("watch_path") or "",
            "dest_path": await agent.get_memory("dest_path") or "",
            "polling_enabled": await agent.get_memory("polling_enabled") or "false",
            "polling_interval": await agent.get_memory("polling_interval") or str(agent.polling_interval),
            "scan_enabled": await agent.get_memory("scan_enabled") or "false",
            "scan_interval": await agent.get_memory("scan_interval") or str(agent.scan_interval)
        }

    @router.post("/config")
    async def update_config(config: ConfigRequest):
        await agent.save_memory("watch_path", config.watch_path)
        await agent.save_memory("dest_path", config.dest_path)
        
        # Restart watcher with new paths
        await agent.start_watching()
        return {"status": "updated"}

    @router.get("/logs")
    async def get_logs():
        return agent.logs

    @router.post("/scan")
    async def trigger_scan():
        await agent.log_to_ui("Manual scan triggered.")
        # Trigger scan in background or await it? 
        # Await for now to keep it simple, though background task is better for long scans
        await agent.scan_folder()
        return {"status": "scanning_started"}

    return router
