from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import io
import os
import time

import pandas as pd
import requests
import yfinance as yf

MAX_BATCH_SIZE = 200

# Get code of tickers.
def get_tickers(source: str = "manual") -> list[str]:
    """
    Return a list of ASX ticker symbols (suffixed with .AX).

    source options:
        "manual"  — hardcoded blue-chip + sector sample list
        "asx200"  — scrape the ASX 200 constituents from Wikipedia
    """
    if source == "manual":
        #  ASX 200 as Mar 2026
        return [
            "A2M.AX", "ABP.AX", "ABC.AX", "ADE.AX", "AD8.AX", "AGL.AX", "AIA.AX", "ALD.AX", "ALL.AX", "ALQ.AX",
            "ALU.AX", "ALX.AX", "AMC.AX", "AMP.AX", "ANN.AX", "ANZ.AX", "APA.AX", "APE.AX", "API.AX", "ARB.AX",
            "ARE.AX", "ARG.AX", "ASB.AX", "AST.AX", "ASX.AX", "AWC.AX", "AZJ.AX", "BAP.AX", "BEN.AX", "BGA.AX",
            "BHP.AX", "BIN.AX", "BKL.AX", "BKW.AX", "BLD.AX", "BOQ.AX", "BPT.AX", "BRG.AX", "BSL.AX", "BWP.AX",
            "BXB.AX", "CAR.AX", "CBA.AX", "CCL.AX", "CCP.AX", "CDA.AX", "CGF.AX", "CHC.AX", "CHN.AX", "CIA.AX",
            "CIM.AX", "CLW.AX", "CMW.AX", "CNU.AX", "COH.AX", "COL.AX", "CPU.AX", "CQR.AX", "CSL.AX", "CSR.AX",
            "CTD.AX", "CWN.AX", "CWY.AX", "DBI.AX", "DMP.AX", "DOW.AX", "DRR.AX", "DXS.AX", "EDV.AX", "ELD.AX",
            "EML.AX", "EVN.AX", "EVT.AX", "FBU.AX", "FLT.AX", "FMG.AX", "FPH.AX", "GMG.AX", "GNE.AX", "GOZ.AX",
            "GPT.AX", "GUD.AX", "GWA.AX", "HLS.AX", "HUB.AX", "HUM.AX", "IAG.AX", "IEL.AX", "IFL.AX", "IFT.AX",
            "IGO.AX", "ILU.AX", "IMU.AX", "INA.AX", "ING.AX", "IPH.AX", "IRE.AX", "IVC.AX", "JBH.AX", "JHX.AX",
            "LLC.AX", "LNK.AX", "LYC.AX", "MFG.AX", "MGR.AX", "MPL.AX", "MQG.AX", "MTS.AX", "NAB.AX", "NAN.AX",
            "NCM.AX", "NEA.AX", "NHF.AX", "NIC.AX", "NSR.AX", "NST.AX", "NUF.AX", "NWL.AX", "NXT.AX", "ORA.AX",
            "ORG.AX", "ORI.AX", "OSH.AX", "OZL.AX", "PBH.AX", "PDN.AX", "PGH.AX", "PLS.AX", "PME.AX", "PMV.AX",
            "PNI.AX", "PNV.AX", "PPT.AX", "PTM.AX", "QAN.AX", "QBE.AX", "QUB.AX", "REA.AX", "REH.AX", "RHC.AX",
            "RIO.AX", "RMD.AX", "RRL.AX", "RSG.AX", "RWC.AX", "S32.AX", "SCG.AX", "SCP.AX", "SDF.AX", "SGM.AX",
            "SGR.AX", "SHL.AX", "SKC.AX", "SKI.AX", "SLK.AX", "SNZ.AX", "SOL.AX", "SPK.AX", "STW.AX", "SUN.AX",
            "SVW.AX", "SYD.AX", "TAH.AX", "TCL.AX", "TLS.AX", "TNE.AX", "TPW.AX", "TPG.AX", "TRN.AX", "TWE.AX",
            "TYR.AX", "VUK.AX", "VCX.AX", "VEA.AX", "VGS.AX", "VTH.AX", "WAM.AX", "WBC.AX", "WEB.AX", "WES.AX",
            "WHC.AX", "WOR.AX", "WOW.AX", "WPL.AX", "WTC.AX", "XRO.AX", "Z1P.AX", "ZEL.AX"
        ]

    try:
        return _get_asx_tickers()
    except Exception as e:
        print(f"[WARNING] Could not fetch listings from ASX: {e}")

    print("[INFO] Falling back to manual list")
    return get_tickers("manual")

def _get_asx_tickers() -> list[str]:
    """
    Return the list of all ASX listed ticker symbols (suffixed with .AX).
    """

    # Use local csv if exists.
    if os.path.exists('ASXListedCompanies.csv'):
        print('Local ASX listings exists. Use local one.')
        df = pd.read_csv('ASXListedCompanies.csv', skiprows=2)
    else:
        url = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"

        headers = {
            'Content-Type': 'text/csv'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data, skiprows=2)
        else:
            print(f"Failed to request for ASX listed companies from ASX with {response.status_code}: {response.text}")
            raise RuntimeError(f"Failed to fetch tickers (status {response.status_code}: {response.text})")
    
    return (df['ASX code'] + '.AX').tolist()

# Get tickers info
def bulk_download_prices(tickers: list[str], period_days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV history for all tickers in a single bulk request.
    Returns {ticker: DataFrame(Open, High, Low, Close, Volume)}.
    """
    end = datetime.today()
    start = end - timedelta(days=period_days)
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    result: dict[str, pd.DataFrame] = {}

    print(f"  -> Bulk downloading {len(tickers)} tickers via yf.Tickers in batches...")

    # Download in batches
    total = len(tickers)
    offset = 0
    remaining = total
    while remaining > 0:
        batchSize = min(remaining, MAX_BATCH_SIZE)
        tickersInBatch = tickers[offset:(offset + batchSize)]
        tickers_obj = yf.Tickers(tickersInBatch)
        raw = tickers_obj.history(
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
        )

        # history() returns a MultiIndex DataFrame with columns (field, ticker);
        # slice each ticker out with xs() to get a plain single-level DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickersInBatch:
                try:
                    tickerDf = raw.xs(ticker, axis=1, level=1)
                    df = tickerDf[ohlcv].dropna(how="all")
                    if len(df) >= 60:
                        result[ticker] = df
                    # else:
                        # print(f'{ticker} has less than 60 days of data. Only {len(df)}')
                except KeyError:
                    print(f'{ticker} was not returned from tickers.history.')
                    pass
        elif len(tickers) == 1 and not raw.empty:
            # Single ticker edge case: no MultiIndex, columns are plain field names
            result[tickers[0]] = raw[ohlcv].dropna(how="all")
        
        #  Move to next batch
        offset += batchSize
        remaining -= batchSize

    print(f"  -> Price data retrieved for {len(result)}/{len(tickers)} tickers")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Concurrent fundamental info fetch
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_one_info(ticker: str) -> tuple[str, dict]:
    """Fetch .info for a single ticker. Runs inside the thread pool.
    Retries once if Yahoo returns a throttled/empty response (no quoteType).
    """
    for attempt in range(2):
        try:
            info = yf.Ticker(ticker).info
            if info.get("quoteType"):  # present in all real responses, absent when throttled
                return ticker, info
            print(f"  [THROTTLED] {ticker} attempt {attempt + 1}, retrying...", flush=True)
            time.sleep(1)
        except Exception:
            pass
    return ticker, {}


def bulk_fetch_fundamentals(tickers: list[str], max_workers: int = 20) -> dict[str, dict]:
    """
    Fetch yfinance .info dicts for all tickers concurrently.

    Why concurrent instead of serial:
      - Each .info call is an independent HTTP request; serial total = N x round-trip latency
      - Concurrent total ≈ one round-trip (threads wait on I/O in parallel)
      - max_workers=20 is a safe empirical limit — higher values trigger Yahoo rate limits
      - No sleep() needed; the connection pool provides natural throttling
    """
    print(f"  -> Fetching fundamentals for {len(tickers)} tickers ({max_workers} threads)...")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one_info, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker, info = future.result()
            results[ticker] = info
            done += 1
            if done % 20 == 0 or done == len(tickers):
                print(f"  -> Fundamentals progress: {done}/{len(tickers)}", end="\r", flush=True)

    print(f"\n  -> Fundamentals fetch complete")
    return results