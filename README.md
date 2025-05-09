# TraderBot
Build a trader bot which looks at sentiment of live news events and trades appropriately.

# Lorentzian Strategy Optimizer
This project includes an optimized Lorentzian Classification trading strategy with parameter optimization capabilities.

## Lorentzian Strategy Files
- `lorentzian_strategy.py` - Original Lorentzian strategy implementation
- `lorentzian_optimized_strategy.py` - Optimized version with improved parameters
- `basic_optimizer.py` - Parameter optimization tool for the Lorentzian strategy

## How to Run the Optimizer

### 1. Install Required Dependencies
```bash
python -m pip install deap tqdm matplotlib
```

### 2. Run the Parameter Optimizer
```bash
python basic_optimizer.py
```
This will test different parameter combinations and save the results to the `optimization_results` directory.

### 3. Apply Optimized Parameters
The optimizer automatically creates an optimized strategy file with the best parameters. You can run it directly:
```bash
python lorentzian_optimized_strategy.py
```

## Parameter Configurations
The optimizer tests several parameter configurations:

1. **Base Configuration** - The original parameters
2. **Optimized Configuration** - Best overall balance of returns and risk
3. **Aggressive Configuration** - Higher returns but more volatility
4. **Conservative Configuration** - Lower returns but less drawdown
5. **Fast RSI with Volatility Filter** - Responsive RSI with volatility protection
6. **All Filters Enabled** - Maximum protection with all filters active
7. **Trend Following** - Optimized for trending markets
8. **Range Trading** - Optimized for ranging markets

## Optimization Tips
- For longer backtests, modify the date range in `basic_optimizer.py`
- To add custom parameter sets, edit the `parameter_sets` list in `basic_optimizer.py`
- Set `force_signals: True` to ensure trades occur despite data limitations
- Use `position_size` to control risk exposure (0.1 = 10% of cash per trade)

## See it live and in action 📺
<img src="https://i.imgur.com/FaQH8rz.png"/>

# Startup 🚀
1. Create a virtual environment `conda create -n trader python=3.10` 
2. Activate it `conda activate trader`

## Important: Package Installation
When installing packages, **always use `pip3` or `python -m pip` instead of `pip`**. This ensures packages are installed in your conda environment rather than your system Python.

3. Install initial deps `pip3 install lumibot timedelta alpaca-trade-api==3.1.1`
4. Install transformers and friends `pip3 install torch torchvision torchaudio transformers`
5. Update the `API_KEY` and `API_SECRET` with values from your Alpaca account 
6. Run the bot `python tradingbot.py`

<p>N.B. Torch installation instructions will vary depending on your operating system and hardware. See here for more: 
<a href="pytorch.org/">PyTorch Installation Instructions</a></p>

If you're getting an SSL error when you attempt to call out to the Alpaca Trading api, you'll need to install the required SSL certificates into your machine.
1. Download the following intermediate SSL Certificates, these are required to communicate with Alpaca
* https://letsencrypt.org/certs/lets-encrypt-r3.pem 
* https://letsencrypt.org/certs/isrg-root-x1-cross-signed.pem 
2. Once downloaded, change the file extension of each file to `.cer` 
3. Double click the file and run through the wizard to install it, use all of the default selections. 

</br>
# Other References 🔗

<p>-<a href="github.com/Lumiwealth/lumibot)">Lumibot</a>:trading bot library, makes lifecycle stuff easier .</p>

# Who, When, Why?

👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.x<br />
📜 License: This project is licensed under the MIT License </br>
