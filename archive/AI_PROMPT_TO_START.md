# Initial Prompt for Next AI Assistant

Copy and paste this when starting the new repository:

---

I'm building a **web application for AI-powered stock analysis**. I already have the ML components built in a separate trading bot repository. I need you to help me create a full-stack web application that leverages these components.

## What I Have (In Separate Repo)

1. **Lorentzian Classifier** - ML-based trading signal generator using Lorentzian distance metric
2. **FinBERT Sentiment Analysis** - Financial sentiment from news headlines using transformer model
3. **Parameter Optimization System** - Tested and optimized on multiple symbols

## What I Need You to Build

A web application where users can:
1. Enter a stock symbol (e.g., "AAPL")
2. Get comprehensive AI analysis including:
   - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
   - Chart pattern recognition
   - Lorentzian ML buy/sell signals
   - FinBERT sentiment analysis from news
   - Entry/exit recommendations with specific price targets
3. View interactive charts with visual indicators
4. See all results in a beautiful dashboard

## Tech Stack

**Backend:** FastAPI (Python)
- Import and use my existing classifier.py and finbert_utils.py
- Fetch stock data using yfinance
- Fetch news for sentiment analysis
- Generate comprehensive analysis
- Cache results in Redis

**Frontend:** React + Next.js
- TradingView charts or Recharts
- Tailwind CSS + shadcn/ui
- Real-time updates
- Mobile responsive

**Deployment:** Railway/Heroku + Vercel

## Priority: MVP in 4 Weeks

Build:
✅ Single symbol analysis
✅ All ML components integrated
✅ Interactive charts
✅ Entry/exit recommendations
✅ Clean, professional UI

**DO NOT BUILD:**
❌ Reddit scraping or social media features
❌ Account aggregation
❌ Broker integrations

## Project Structure I Want

```
backend/              # FastAPI application
├── app/
│   ├── api/routes/   # API endpoints
│   ├── core/          # ML components (import mine)
│   ├── utils/         # Data fetching, caching
│   └── models/        # Pydantic models

frontend/             # React application
├── src/
│   ├── components/    # UI components
│   ├── pages/         # Pages/routes
│   └── lib/          # API client, utils
```

## Legal Requirement

**Critical:** All analysis must have disclaimers. Frame as "educational analysis" not "buy/sell recommendations." Use language like "patterns suggest" not "you should buy."

## Your First Tasks

1. Set up the backend FastAPI project structure
2. Create the main API endpoint: `POST /api/analyze/{symbol}`
3. Integrate yfinance for stock data fetching
4. Show me a working example with AAPL

Once that's working, we'll build the frontend and integrate my existing ML components.

Let's start with the backend foundation. What's your plan?

