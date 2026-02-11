"""
Quick run script for the Green Bond Classification API
"""
import uvicorn

if __name__ == "__main__":
    print("="*60)
    print("GREEN BOND CLASSIFICATION API")
    print("   Bank Indonesia - DSta-DSMF")
    print("="*60)
    print("\nServer running at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
