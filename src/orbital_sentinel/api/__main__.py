"""Run the API server: python -m orbital_sentinel.api"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "orbital_sentinel.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
