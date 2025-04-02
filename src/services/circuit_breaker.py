import asyncio
import logging
import time
from enum import Enum
from typing import Callable, TypeVar, Generic

T = TypeVar('T')


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Not allowing requests
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker(Generic[T]):
    def __init__(
            self,
            name: str,
            failure_threshold: int = 5,
            recovery_timeout: int = 30,
            timeout_seconds: int = 10
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout_seconds = timeout_seconds

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger(f"circuit_breaker.{name}")

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute the function with circuit breaker protection"""
        self._check_state_transition()

        if self.state == CircuitState.OPEN:
            self.logger.warning(f"Circuit {self.name} is OPEN - fast failing")
            raise Exception(f"Service {self.name} is unavailable (circuit open)")

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Success, reset failure count if in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.logger.info(f"Circuit {self.name} recovered, moving to CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0

            return result

        except Exception as e:
            self._handle_failure(e)
            raise

    def _check_state_transition(self):
        """Check if the circuit breaker should change state based on timing"""
        if (self.state == CircuitState.OPEN and
                time.time() - self.last_failure_time > self.recovery_timeout):
            self.logger.info(f"Circuit {self.name} recovery timeout elapsed, moving to HALF_OPEN")
            self.state = CircuitState.HALF_OPEN

    def _handle_failure(self, exception):
        """Handle a failure by incrementing counters and potentially opening the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.logger.warning(f"Circuit {self.name} failed in HALF_OPEN state, moving back to OPEN")
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.logger.warning(
                f"Circuit {self.name} reached failure threshold ({self.failure_count}), moving to OPEN"
            )
            self.state = CircuitState.OPEN
