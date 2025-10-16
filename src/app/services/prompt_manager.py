import json
import logging
import os
import time
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)


class PromptTemplate:
    def __init__(
            self,
            id: str,
            name: str,
            version: str,
            template: str,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.name = name
        self.version = version
        self.template = template
        self.description = description
        self.metadata = metadata or {}

    def format(self, **kwargs) -> str:
        """Format the template with the provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template: {e}")
            raise ValueError(f"Missing required variable: {e}")
        except Exception as e:
            logger.error(f"Error formatting prompt template: {e}")
            raise


class PromptManager:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the prompt manager

        Args:
            api_key: API key for external prompt management service (if used)
            base_url: Base URL for external prompt management service (if used)
        """
        self.api_key = api_key or os.environ.get("PROMPT_API_KEY")
        self.base_url = base_url or os.environ.get("PROMPT_API_URL")
        self.local_templates: Dict[str, PromptTemplate] = {}
        self.use_external_service = bool(self.api_key and self.base_url)

        # Load local templates
        self._load_local_templates()

    def _load_local_templates(self):
        """Load prompt templates from local storage"""
        template_dir = os.environ.get("PROMPT_TEMPLATE_DIR", "./prompts")
        if not os.path.exists(template_dir):
            logger.warning(f"Prompt template directory not found: {template_dir}")
            return

        try:
            for filename in os.listdir(template_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(template_dir, filename), "r") as f:
                        template_data = json.load(f)
                        template = PromptTemplate(
                            id=template_data.get("id", filename.replace(".json", "")),
                            name=template_data.get("name", ""),
                            version=template_data.get("version", "1.0"),
                            template=template_data.get("template", ""),
                            description=template_data.get("description", ""),
                            metadata=template_data.get("metadata", {})
                        )
                        self.local_templates[template.id] = template
                        logger.debug(f"Loaded prompt template: {template.id} (v{template.version})")

            logger.info(f"Loaded {len(self.local_templates)} prompt templates from {template_dir}")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")

    async def get_template(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a prompt template by ID

        Args:
            template_id: ID of the template to retrieve
            version: Specific version to retrieve (optional)

        Returns:
            PromptTemplate: The requested prompt template
        """
        # Check if we should use external service
        if self.use_external_service:
            return await self._get_remote_template(template_id, version)

        # Otherwise use local templates
        if template_id in self.local_templates:
            template = self.local_templates[template_id]
            if version and template.version != version:
                logger.warning(f"Template version mismatch: requested {version}, got {template.version}")
            return template

        raise ValueError(f"Prompt template not found: {template_id}")

    async def _get_remote_template(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a prompt template from the remote service"""
        if not self.api_key or not self.base_url:
            raise ValueError("External prompt service not configured")

        url = f"{self.base_url}/templates/{template_id}"
        if version:
            url += f"?version={version}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=5.0)
                response.raise_for_status()
                data = response.json()

                return PromptTemplate(
                    id=data.get("id", template_id),
                    name=data.get("name", ""),
                    version=data.get("version", "1.0"),
                    template=data.get("template", ""),
                    description=data.get("description", ""),
                    metadata=data.get("metadata", {})
                )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error retrieving prompt template: {e.response.status_code} - {e.response.text}")
            # Fall back to local template if available
            if template_id in self.local_templates:
                logger.info(f"Falling back to local template: {template_id}")
                return self.local_templates[template_id]
            raise ValueError(f"Failed to retrieve template: {template_id}")
        except Exception as e:
            logger.error(f"Error retrieving prompt template: {e}")
            # Fall back to local template if available
            if template_id in self.local_templates:
                logger.info(f"Falling back to local template: {template_id}")
                return self.local_templates[template_id]
            raise

    async def log_prompt_use(
            self,
            template_id: str,
            variables: Dict[str, Any],
            result_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log usage of a prompt template to the remote service"""
        if not self.use_external_service:
            logger.debug(f"Prompt use logged locally: {template_id}")
            return

        url = f"{self.base_url}/templates/{template_id}/usage"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "timestamp": int(time.time()),
            "variables": variables,
            "result_metrics": result_metrics or {}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data, timeout=5.0)
                response.raise_for_status()
                logger.debug(f"Prompt use logged to remote service: {template_id}")
        except Exception as e:
            logger.warning(f"Failed to log prompt usage: {e}")

    async def enhance_system_prompt(
            self,
            original_prompt: str,
            context: Dict[str, Any]
    ) -> str:
        """Enhance a system prompt with additional capabilities based on context

        Args:
            original_prompt: The original system prompt
            context: Context information for enhancement

        Returns:
            str: Enhanced system prompt
        """
        try:
            # Get template for enhancing system prompts
            template = await self.get_template("system_prompt_enhancer")

            # Format template with original prompt and context
            enhanced_params = {
                "original_prompt": original_prompt,
                **context
            }

            enhanced_prompt = template.format(**enhanced_params)
            return enhanced_prompt
        except Exception as e:
            logger.warning(f"Failed to enhance system prompt: {e}")
            return original_prompt
