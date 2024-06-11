from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()


# Define a path operation decorator for the root endpoint
# This decorator registers the 'root' function to handle GET requests at the root URL ("/")
@app.get("/")
async def root():
    # Define an asynchronous function 'root' to handle requests at the root URL
    # This function returns a JSON response with a message
    return {"message": "Testing...FastAPI functional."}

# In terminal, use uvicorn main:app --reload
