import time
import threading
from typing import Optional

class RateLimiter:
    """
    Thread-safe rate limiter for RPM (Requests Per Minute) and TPM (Tokens Per Minute).
    Uses a token bucket algorithm.
    """
    def __init__(self, rpm_limit: Optional[int] = None, tpm_limit: Optional[int] = None):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # RPM state
        self.request_tokens = float(rpm_limit) if rpm_limit else float('inf')
        self.last_request_refill = time.time()
        
        # TPM state
        self.token_tokens = float(tpm_limit) if tpm_limit else float('inf')
        self.last_token_refill = time.time()
        
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        
        # Refill RPM
        if self.rpm_limit:
            elapsed = now - self.last_request_refill
            refill_amount = elapsed * (self.rpm_limit / 60.0)
            self.request_tokens = min(self.rpm_limit, self.request_tokens + refill_amount)
            self.last_request_refill = now
            
        # Refill TPM
        if self.tpm_limit:
            elapsed = now - self.last_token_refill
            refill_amount = elapsed * (self.tpm_limit / 60.0)
            self.token_tokens = min(self.tpm_limit, self.token_tokens + refill_amount)
            self.last_token_refill = now

    def acquire(self, tokens: int = 0):
        """
        Blocks until permission is granted to make a request.
        :param tokens: Estimated number of tokens this request will consume (input + output).
                       If unknown, use a conservative estimate or 0 (only RPM limiting).
        """
        if not self.rpm_limit and not self.tpm_limit:
            return

        while True:
            with self.lock:
                self._refill()
                
                can_request = True
                if self.rpm_limit and self.request_tokens < 1:
                    can_request = False
                
                if self.tpm_limit and self.token_tokens < tokens:
                    can_request = False
                
                if can_request:
                    if self.rpm_limit:
                        self.request_tokens -= 1
                    if self.tpm_limit:
                        self.token_tokens -= tokens
                    return
            
            # Wait a bit before checking again
            time.sleep(0.1)
