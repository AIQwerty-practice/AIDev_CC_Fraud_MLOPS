"""Environment-aware H2O connection helpers shared by training and serving."""

import os

import h2o


def connect_h2o():
    """Connect to a shared H2O service, or start a local cluster as fallback.

    Returns True when the cluster is externally managed and must not be shut down
    by the current process.
    """
    service_url = os.getenv("INTERNAL_H2O_URL") or os.getenv("H2O_URL")
    if service_url:
        h2o.connect(url=service_url)
        return True
    h2o.init()
    return False
