from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.feedback import router as feedback_router
from api.routes.score import router as score_router

app = FastAPI(title="Credit Sense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")


@app.get("/")
def root() -> dict:
    return {"status": "Credit Sense API is running"}
