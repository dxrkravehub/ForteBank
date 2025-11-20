from fastapi import FastAPI

app = FastAPI(title="Zaglushka")
@app.get('/')
def root():
    return {"message": "Hello from fastAPI!"}

@app.get('/health')
def health():
    return {"status":"ok"}