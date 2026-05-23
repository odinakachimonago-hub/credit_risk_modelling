# AI Credit Risk Decision Engine

## Backend
cd backend
pip3 install -r requirements.txt
python3 -m uvicorn app.main:app --reload --port 9002

## Frontend
cd frontend
npm install
npm run dev -- --port 3002

Open http://localhost:3002
