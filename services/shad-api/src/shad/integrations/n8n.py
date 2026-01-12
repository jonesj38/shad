"""n8n Integration - Webhook triggers and workflow orchestration.

Per SPEC.md Appendix A, n8n serves as:
- Thin trigger + automation layer (not orchestration peer)
- Scheduling (cron)
- Event wiring (webhooks)
- Fan-out/fan-in of runs (not nodes)
- Integrations (Slack, email, etc.)

n8n workflow pattern:
1. Start run
2. Wait for completion webhook
3. Fetch final.summary.json
4. Route by status (complete → publish, partial → review, failed → alert)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel

from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Events that trigger n8n webhooks."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_PARTIAL = "run.partial"
    RUN_FAILED = "run.failed"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    REVIEW_NEEDED = "review.needed"


class WebhookPayload(BaseModel):
    """Payload sent to n8n webhooks."""

    event: WebhookEvent
    run_id: str
    timestamp: str
    status: str | None = None
    goal: str | None = None
    result: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    @classmethod
    def from_run(cls, event: WebhookEvent, run: Any) -> WebhookPayload:
        """Create payload from a Run object."""
        return cls(
            event=event,
            run_id=run.run_id,
            timestamp=datetime.utcnow().isoformat(),
            status=run.status.value if hasattr(run.status, "value") else str(run.status),
            goal=run.config.goal,
            result=run.final_result,
            error=run.error,
            metrics={
                "total_nodes": len(run.nodes),
                "total_tokens": run.total_tokens,
            },
        )


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""

    url: str
    events: list[WebhookEvent]
    secret: str | None = None
    enabled: bool = True
    retry_count: int = 3
    timeout_seconds: int = 30
    headers: dict[str, str] = field(default_factory=dict)


class N8NClient:
    """
    Client for n8n webhook integration.

    Handles:
    - Webhook registration and dispatch
    - Event filtering and routing
    - Retry logic for failed deliveries
    - Payload signing for security
    """

    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = base_url or getattr(settings, "n8n_base_url", None)
        self.webhooks: list[WebhookConfig] = []
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
            )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def register_webhook(
        self,
        url: str,
        events: list[WebhookEvent],
        secret: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Register a webhook endpoint.

        Args:
            url: Webhook URL to call
            events: List of events that trigger this webhook
            secret: Optional secret for payload signing
            **kwargs: Additional WebhookConfig options
        """
        config = WebhookConfig(
            url=url,
            events=events,
            secret=secret,
            **kwargs,
        )
        self.webhooks.append(config)
        logger.info(f"Registered webhook for events {events} at {url}")

    async def dispatch(self, payload: WebhookPayload) -> dict[str, bool]:
        """
        Dispatch a webhook event to all registered endpoints.

        Returns dict mapping webhook URLs to success status.
        """
        if not self._http_client:
            await self.connect()

        results: dict[str, bool] = {}

        for webhook in self.webhooks:
            if not webhook.enabled:
                continue

            if payload.event not in webhook.events:
                continue

            success = await self._send_webhook(webhook, payload)
            results[webhook.url] = success

        return results

    async def _send_webhook(
        self,
        webhook: WebhookConfig,
        payload: WebhookPayload,
    ) -> bool:
        """Send payload to a single webhook with retry logic."""
        if not self._http_client:
            return False

        headers = {
            "Content-Type": "application/json",
            **webhook.headers,
        }

        # Add signature if secret configured
        if webhook.secret:
            import hashlib
            import hmac

            payload_bytes = payload.model_dump_json().encode()
            signature = hmac.new(
                webhook.secret.encode(),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Shad-Signature"] = signature

        for attempt in range(webhook.retry_count):
            try:
                response = await self._http_client.post(
                    webhook.url,
                    json=payload.model_dump(),
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                if response.status_code < 300:
                    logger.debug(f"Webhook delivered to {webhook.url}")
                    return True

                logger.warning(
                    f"Webhook to {webhook.url} returned {response.status_code}"
                )

            except Exception as e:
                logger.warning(
                    f"Webhook to {webhook.url} failed (attempt {attempt + 1}): {e}"
                )

            # Exponential backoff between retries
            if attempt < webhook.retry_count - 1:
                import asyncio

                await asyncio.sleep(2**attempt)

        logger.error(f"Webhook to {webhook.url} failed after {webhook.retry_count} attempts")
        return False

    async def notify_run_started(self, run: Any) -> None:
        """Convenience method to notify run start."""
        payload = WebhookPayload.from_run(WebhookEvent.RUN_STARTED, run)
        await self.dispatch(payload)

    async def notify_run_completed(self, run: Any) -> None:
        """Convenience method to notify run completion."""
        event = WebhookEvent.RUN_COMPLETED
        if hasattr(run, "status"):
            status = run.status.value if hasattr(run.status, "value") else str(run.status)
            if status == "partial":
                event = WebhookEvent.RUN_PARTIAL
            elif status == "failed":
                event = WebhookEvent.RUN_FAILED

        payload = WebhookPayload.from_run(event, run)
        await self.dispatch(payload)

    async def trigger_workflow(
        self,
        workflow_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Trigger an n8n workflow via HTTP.

        Args:
            workflow_id: n8n workflow ID or webhook path
            data: Data to send to the workflow

        Returns:
            Workflow response or None if failed
        """
        if not self.base_url:
            logger.warning("n8n base URL not configured")
            return None

        if not self._http_client:
            await self.connect()

        url = f"{self.base_url}/webhook/{workflow_id}"

        try:
            response = await self._http_client.post(
                url,
                json=data or {},
                timeout=60.0,
            )

            if response.status_code < 300:
                return response.json()

            logger.warning(f"n8n workflow trigger failed: {response.status_code}")

        except Exception as e:
            logger.error(f"n8n workflow trigger error: {e}")

        return None


# Webhook handler for incoming n8n triggers
class WebhookHandler:
    """
    Handles incoming webhooks from n8n.

    Used to trigger Shad runs from n8n workflows.
    """

    def __init__(self, secret: str | None = None):
        self.secret = secret
        self.handlers: dict[str, Any] = {}

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not self.secret:
            return True

        import hashlib
        import hmac

        expected = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def register_handler(self, event_type: str, handler: Any) -> None:
        """Register a handler for an event type."""
        self.handlers[event_type] = handler

    async def handle(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle an incoming webhook."""
        handler = self.handlers.get(event_type)
        if not handler:
            return {"error": f"No handler for event type: {event_type}"}

        try:
            result = await handler(payload)
            return {"success": True, "result": result}
        except Exception as e:
            logger.exception(f"Webhook handler error: {e}")
            return {"error": str(e)}
