# AI Stock Analysis Web Application - Project Briefing

## Project Overview

You are helping build a **web-based AI stock analysis application** that allows users to input a stock symbol and receive comprehensive technical analysis, sentiment analysis, and entry/exit strategy recommendations.

### Core Value Proposition
The application combines advanced machine learning classification (Lorentzian Classification) with financial sentiment analysis (FinBERT) to provide users with actionable trading insights. This is a **production-ready web application** to be deployed as a portfolio showcase and potential SaaS product.

---

## Technical Context

### Existing Components (Already implemented)

The developer has an existing trading bot project with these components:

1. **Lorentzian Classifier** (`classifier.py`)
   - ML classification using Lorentzian distance metric (alternative to Euclidean)
   - Accounts for market "warping effects" from economic events
   - Features: RSI, WT (WaveTrend), CCI, ADX with normalization
   - Includes kernel regression filters and regime detection
   - Currently designed for Lumibot trading strategies

2. **FinBERT Sentiment Analysis** (`finbert_utils.py`)
   - Uses ProsusAI/finbert transformer model
   - Analyzes financial news headlines
   - Returns sentiment: positive/negative/neutral with confidence scores
   - Already functional and tested

3. **Parameter Optimization System**
   - Smart optimization with state persistence
   - Multi-strategy approach (random exploration, local search, genetic algorithms)
   - Saves best parameters to JSON files
   - Backtested on multiple symbols (SPY, TSLA, PLTR, AMPX)

### Project Location
This is a **NEW separate repository** for the web application. The existing trading bot code should be **imported or adapted** but this is a standalone project.

---

## Features to Build

### Feature 1: Symbol Analysis Dashboard (MVP)

**User Flow:**
1. User enters a stock symbol (e.g., "AAPL")
2. System fetches data and runs analysis
3. Displays comprehensive analysis dashboard

**What to Analyze:**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages (50/200)
- **Chart Pattern Recognition**: Head & shoulders, double tops/bottoms, triangles, flags
- **Lorentzian ML Classification**: Buy/sell signals using existing classifier
- **Sentiment Analysis**: FinBERT analysis of recent news headlines
- **Entry/Exit Recommendations**: Specific price targets based on analysis

**Required Outputs:**
```json
{
  "symbol": "AAPL",
  "price": 177.50,
  "indicators": {
    "rsi": 68,
    "macd": "bullish_crossover",
    "bb_position": "upper_band",
    "ma_50": 175.20,
    "ma_200": 170.80,
    "trend": "bullish"
  },
  "patterns": [
    {
      "type": "bullish_flag",
      "confidence": 0.75,
      "description": "Pattern suggests upward continuation"
    }
  ],
  "sentiment": {
    "overall": "positive",
    "confidence": 0.82,
    "news_count": 10,
    "recent_headlines": ["Apple announces...", "New product line..."]
  },
  "ml_signal": {
    "direction": "buy",
    "strength": 7,
    "confidence": 0.85,
    "reasoning": "Multiple bullish indicators align"
  },
  "recommendations": {
    "entry_price": [175.0, 178.0],
    "stop_loss": 170.0,
    "take_profit": 185.0,
    "risk_reward": "1:1.25",
    "notes": "Golden cross detected, momentum building"
  }
}
```

### Feature 2: Interactive Charts (High Priority)

**Visualization Requirements:**
- Candlestick chart with OHLC data
- Overlay technical indicators:
  - Moving Averages (50, 200)
  - Bollinger Bands
  - Volume bars
- Mark entry/exit points on chart
- Support multiple timeframes (1D, 1W, 1M)

**Libraries to Consider:**
- TradingView Lightweight Charts (preferred)
- Recharts (alternative)
- Chart.js with plugins
- Plotly

### Feature 3: Historical Backtesting View (Phase 2)

**Concept:**
Show users how the analysis would have performed historically.

**Features:**
- Select date range
- See what signals would have been generated
- Visualize equity curve
- Compare against buy & hold

### Feature 4: Multi-Strategy Support (Phase 3)

**Concept:**
Allow users to see analysis from multiple strategies:
- Lorentzian (default)
- MA Cross Strategy
- Custom user parameters

**UI:** Tabs or dropdown to switch between strategies

---

## Technical Architecture

### Tech Stack

**Backend:**
- **Framework**: FastAPI (Python)
- **ML Models**: 
  - Import existing `classifier.py` and `finbert_utils.py`
  - Add technical indicator calculations using `pandas-ta`
- **Data Sources**:
  - `yfinance` for stock data (free, reliable)
  - `Alpha Vantage` or `NewsAPI` for financial news
  - Optionally: Polygon.io for professional-grade data
- **Database**: 
  - PostgreSQL for user data, analysis history
  - Redis for caching (OHLC data, API responses)
- **Authentication**: JWT tokens (later phases)

**Frontend:**
- **Framework**: React + Next.js (or Vite)
- **Styling**: Tailwind CSS + shadcn/ui components
- **Charting**: TradingView Lightweight Charts or Recharts
- **State Management**: React Query / SWR for data fetching
- **WebSockets**: For real-time price updates (optional)

**Deployment:**
- **Backend**: Railway, Render, or AWS Lambda
- **Frontend**: Vercel or Netlify
- **Database**: Supabase, Neon, or Railway Postgres
- **Containerization**: Docker (optional but recommended)

### Project Structure

```
stock-analysis-app/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── analyze.py
│   │   │   │   ├── history.py
│   │   │   │   └── charts.py
│   │   │   ├── core/
│   │   │   │   ├── classifier.py         # Import from trading bot
│   │   │   │   ├── finbert.py            # Import from trading bot
│   │   │   │   ├── indicators.py         # Technical indicators
│   │   │   │   ├── patterns.py          # Pattern recognition
│   │   │   │   └── recommendations.py    # Entry/exit logic
│   │   ├── models/
│   │   │   └── analysis.py               # Pydantic models
│   │   └── utils/
│   │       ├── data_fetcher.py          # yfinance wrapper
│   │       ├── cache.py                  # Redis caching
│   │       └── validators.py
│   ├── requirements.txt
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalysisDashboard.jsx
│   │   │   ├── StockChart.jsx
│   │   │   ├── IndicatorsPanel.jsx
│   │   │   ├── SentimentAnalysis.jsx
│   │   │   ├── RecommendationBox.jsx
│   │   │   └── SymbolInput.jsx
│   │   ├── pages/
│   │   │   ├── index.jsx
│   │   │   └── history.jsx
│   │   ├── lib/
│   │   │   ├── api.js
│   │   │   └── utils.js
│   │   └── App.jsx
│   ├── package.json
│   └── tailwind.config.js
│
├── README.md
└── .env.example

```

---

## Implementation Roadmap

### Week 1: Backend Foundation
**Goals:**
- Set up FastAPI project structure
- Integrate existing ML components
- Create data fetching pipeline (yfinance)
- Implement basic indicators (RSI, MACD, MA)
- Build first API endpoint: `POST /analyze/{symbol}`

**Deliverable:** Working API that returns basic analysis for AAPL

### Week 2: Core Analysis Engine
**Goals:**
- Integrate Lorentzian classifier
- Integrate FinBERT sentiment analysis
- Add chart pattern recognition
- Generate entry/exit recommendations
- Cache responses in Redis

**Deliverable:** Full analysis pipeline running

### Week 3: Frontend Dashboard
**Goals:**
- Set up React project
- Build analysis dashboard UI
- Integrate charting library
- Display all analysis components
- Add loading states and error handling

**Deliverable:** Functional web app UI

### Week 4: Polish & Deploy
**Goals:**
- Add animations and transitions
- Implement caching on frontend
- Add tooltips and explanations
- Deploy backend and frontend
- Write documentation

**Deliverable:** Live deployed application

---

## Important Constraints & Considerations

### Legal/Compliance
- **Never present signals as financial advice**
- Use disclaimers: "For educational purposes only"
- All recommendations should include educational context
- No automated trading or broker integration

### Performance
- Cache stock data (refresh every 15 minutes)
- Cache sentiment analysis (1 hour TTL)
- Use Redis for session management
- Implement request rate limiting

### User Experience
- Fast loading times (< 2 seconds for analysis)
- Clear visual hierarchy (chart is focal point)
- Educational tooltips explaining indicators
- Mobile-responsive design
- Accessible (WCAG 2.1 AA)

### Extensibility
- Design for multiple ML strategies
- Allow parameter tuning (future feature)
- Support custom symbols beyond major exchanges
- Prepare for user authentication (phase 2)

---

## Key Requirements

### API Endpoints (Backend)

```
POST   /api/analyze/{symbol}        # Run full analysis
GET    /api/analysis/{symbol}       # Get cached analysis
GET    /api/history/{symbol}         # Historical price data
GET    /api/indicators/{symbol}     # Just indicators
GET    /api/sentiment/{symbol}      # Just sentiment
GET    /api/chart-data/{symbol}     # OHLC for charts
```

### UI Components (Frontend)

1. **Symbol Input**
   - Search bar with autocomplete
   - Recent searches
   - Popular stocks quick access

2. **Analysis Dashboard**
   - Stock chart (main visual)
   - Indicators grid
   - Sentiment card
   - Pattern detection card
   - Recommendations card

3. **Chart Component**
   - Interactive candlestick
   - Indicator overlays
   - Entry/exit markers
   - Timeframe selector
   - Zoom/pan controls

4. **Recommendations**
   - Entry price range
   - Stop loss
   - Take profit
   - Risk/reward ratio
   - Visual target zones on chart

---

## Sample Implementation

### Example API Response Structure

```python
# app/api/routes/analyze.py
from fastapi import APIRouter, HTTPException
from app.core.indicators import calculate_indicators
from app.core.classifier import run_lorentzian_analysis
from app.core.finbert import analyze_news_sentiment
from app.core.patterns import detect_chart_patterns
from app.utils.data_fetcher import fetch_stock_data

router = APIRouter(prefix="/api")

@router.post("/analyze/{symbol}")
async def analyze_stock(symbol: str):
    try:
        # Fetch data
        df = fetch_stock_data(symbol, period="1y")
        news = fetch_news(symbol)
        
        # Run analyses
        indicators = calculate_indicators(df)
        ml_signal = run_lorentzian_analysis(df)
        sentiment = analyze_news_sentiment(news)
        patterns = detect_chart_patterns(df)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            indicators, ml_signal, sentiment, patterns
        )
        
        return {
            "symbol": symbol,
            "current_price": df['close'].iloc[-1],
            "timestamp": df.index[-1].isoformat(),
            "indicators": indicators,
            "patterns": patterns,
            "sentiment": sentiment,
            "ml_signal": ml_signal,
            "recommendations": recommendations,
            "disclaimer": "Not financial advice. Educational purposes only."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Example UI Component

```jsx
// frontend/src/components/AnalysisDashboard.jsx
import { useState } from 'react';
import StockChart from './StockChart';
import IndicatorsPanel from './IndicatorsPanel';
import SentimentAnalysis from './SentimentAnalysis';
import RecommendationBox from './RecommendationBox';

export default function AnalysisDashboard({ symbol }) {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  
  const fetchAnalysis = async (sym) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/analyze/${sym}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchAnalysis(symbol);
  }, [symbol]);
  
  if (loading) return <LoadingSpinner />;
  if (!analysis) return <ErrorMessage />;
  
  return (
    <div className="dashboard">
      <StockChart 
        symbol={symbol} 
        data={analysis.ohlc}
        entry={analysis.recommendations.entry_price}
        exit={analysis.recommendations.take_profit}
      />
      
      <div className="analysis-grid">
        <IndicatorsPanel indicators={analysis.indicators} />
        <SentimentAnalysis sentiment={analysis.sentiment} />
        <RecommendationBox recommendations={analysis.recommendations} />
      </div>
      
      <Disclaimer />
    </div>
  );
}
```

---

## Success Criteria

### MVP (Weeks 1-4)
- [ ] User can enter any stock symbol
- [ ] System fetches data and runs all analyses
- [ ] Dashboard displays comprehensive analysis
- [ ] Interactive charts with indicators
- [ ] Entry/exit recommendations are shown
- [ ] Application is deployed and accessible
- [ ] Documentation is complete

### Quality Standards
- API response time < 2 seconds
- Mobile-responsive design
- No console errors in production
- Clear educational disclaimers
- Accessible to screen readers
- SEO-friendly metadata

---

## Additional Notes

### Prioritized Features (MVP Only)
Focus on these for the first version:
1. Single symbol analysis (the core feature)
2. Traditional technical indicators (RSI, MACD, MA)
3. Lorentzian ML signals
4. FinBERT sentiment
5. Entry/exit recommendations
6. Basic chart visualization

### Future Enhancements (Post-MVP)
- Multiple strategy comparison
- User accounts and favorites
- Historical performance tracking
- Email alerts
- Mobile app
- Advanced pattern recognition
- Multi-timeframe analysis

### Code Quality
- Use TypeScript where possible
- Follow PEP 8 (Python) and ESLint (JavaScript)
- Write unit tests for core functions
- Add error handling everywhere
- Log important events
- Use environment variables for secrets

---

## Questions to Consider

As you build this, consider:
1. How to make the ML classification run faster (optimization needed?)
2. What if a symbol has insufficient data?
3. How to handle market holidays and weekends?
4. Should we support crypto symbols?
5. How to make recommendations more actionable?
6. What additional data sources might improve accuracy?

---

## Getting Started

1. Ask the user what they want to start with (backend vs frontend)
2. Create the project structure
3. Set up the foundational files
4. Begin with data fetching
5. Build incrementally, testing each component

**Remember:** This is a showcase project that demonstrates full-stack development, ML engineering, and financial domain expertise. Make it production-quality!

---

## Environment Variables Needed

```bash
# Backend (.env)
POLYGON_API_KEY=          # Optional, for premium data
ALPHA_VANTAGE_KEY=        # For news API
NEWS_API_KEY=             # Alternative news source
REDIS_URL=                # For caching
DATABASE_URL=             # PostgreSQL connection

# Frontend (.env)
NEXT_PUBLIC_API_URL=      # Backend URL
```

---

**End of Briefing** - You're ready to build! 🚀
