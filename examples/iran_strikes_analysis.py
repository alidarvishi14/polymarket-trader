"""Analysis of US strikes Iran term structure from Polymarket.

Data source: https://polymarket.com/event/us-strikes-iran-by
Data as of: January 27, 2026
"""

import logging
from datetime import date

import numpy as np

from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.optimization.curve_fitting import HazardCurveFitter, LossType
from polymarket_trader.optimization.simple_portfolio import optimize_simple_portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run term structure analysis on US-Iran strikes market."""
    # Current date
    today = date(2026, 1, 27)

    # Market data from Polymarket (as of Jan 27, 2026)
    # Format: (expiry_date, yes_price, bid, ask, volume, contract_id)
    # Spreads estimated at ~1-2 cents based on typical Polymarket spreads
    raw_data = [
        # (date(2026, 1, 27), 0.013, 0.010, 0.015, 1_672_646, "jan27"),  # Today - skip
        (date(2026, 1, 28), 0.034, 0.030, 0.038, 267_014, "jan28"),
        (date(2026, 1, 29), 0.045, 0.041, 0.049, 183_051, "jan29"),
        (date(2026, 1, 30), 0.080, 0.076, 0.084, 137_756, "jan30"),
        (date(2026, 1, 31), 0.110, 0.105, 0.115, 25_254_997, "jan31"),
        (date(2026, 2, 6), 0.200, 0.190, 0.210, 148_464, "feb06"),
        (date(2026, 2, 13), 0.310, 0.300, 0.320, 122_588, "feb13"),
        (date(2026, 2, 28), 0.440, 0.430, 0.450, 2_732_821, "feb28"),
        (date(2026, 3, 31), 0.550, 0.540, 0.560, 3_629_017, "mar31"),
        (date(2026, 6, 30), 0.650, 0.640, 0.660, 1_958_185, "jun30"),
    ]

    # Convert to MaturityData (days to expiry)
    market_data = []
    for expiry, price, bid, ask, volume, contract_id in raw_data:
        days_to_expiry = (expiry - today).days
        if days_to_expiry > 0:
            market_data.append(
                MaturityData(
                    maturity=float(days_to_expiry),
                    market_price=price,
                    bid_price=bid,
                    ask_price=ask,
                    volume=volume,
                    contract_id=contract_id,
                )
            )

    print("=" * 70)
    print("US STRIKES IRAN - TERM STRUCTURE ANALYSIS")
    print("=" * 70)
    print(f"\nData as of: {today}")
    print(f"Number of contracts: {len(market_data)}")

    # Display raw market data
    print("\n" + "-" * 70)
    print("RAW MARKET DATA")
    print("-" * 70)
    print(f"{'Contract':<10} {'DTE':>6} {'Price':>8} {'Bid':>8} {'Ask':>8} {'Volume':>15}")
    print("-" * 70)
    for d in market_data:
        print(
            f"{d.contract_id:<10} {d.maturity:>6.0f} {d.market_price:>8.1%} "
            f"{d.bid_price:>8.1%} {d.ask_price:>8.1%} ${d.volume:>14,.0f}"
        )

    # ==========================================================================
    # STEP 1: Fit the hazard curve
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: HAZARD CURVE FITTING")
    print("=" * 70)

    # Try multiple smoothness values to assess model stability
    alphas = [0.01, 0.1, 0.5, 1.0]
    results = {}

    for alpha in alphas:
        fitter = HazardCurveFitter(
            smoothness_alpha=alpha,
            loss_type=LossType.QUADRATIC,
        )
        results[alpha] = fitter.fit(market_data)

    # Use moderate smoothness for main analysis
    main_alpha = 0.1
    fit_result = results[main_alpha]
    model = fit_result.model

    print(f"\nUsing smoothness alpha = {main_alpha}")
    print(f"Solver status: {fit_result.solver_status}")
    print(f"Objective value: {fit_result.objective_value:.6f}")

    # Display fitted hazard curve
    print("\n" + "-" * 70)
    print("FITTED HAZARD CURVE")
    print("-" * 70)
    print(
        f"{'DTE':>6} {'Mkt Price':>10} {'Theo Price':>10} {'Mispricing':>10} "
        f"{'Cum Hazard':>10} {'Bucket λ':>10}"
    )
    print("-" * 70)

    maturities = model.maturities
    theo_prices = model.theoretical_prices
    cum_hazards = model.cumulative_hazards
    bucket_hazards = model.bucket_hazards
    market_prices = np.array([d.market_price for d in market_data])
    mispricings = model.mispricings(market_prices)

    for i in range(len(maturities)):
        print(
            f"{maturities[i]:>6.0f} {market_prices[i]:>10.2%} {theo_prices[i]:>10.2%} "
            f"{mispricings[i]:>+10.2%} {cum_hazards[i]:>10.4f} {bucket_hazards[i]:>10.4f}"
        )

    # ==========================================================================
    # STEP 2: Check for arbitrage / mispricing stability
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: MISPRICING STABILITY ACROSS SMOOTHNESS VALUES")
    print("=" * 70)
    print(f"{'Contract':<10}", end="")
    for alpha in alphas:
        print(f"{'α=' + str(alpha):>10}", end="")
    print(f"{'Stable?':>10}")
    print("-" * 70)

    for i, d in enumerate(market_data):
        print(f"{d.contract_id:<10}", end="")
        misps = []
        for alpha in alphas:
            m = results[alpha].model
            mp = np.array([x.market_price for x in market_data])
            misp = m.mispricings(mp)[i]
            misps.append(misp)
            print(f"{misp:>+10.2%}", end="")

        # Check stability (all same sign and similar magnitude)
        signs_same = all(m > 0 for m in misps) or all(m < 0 for m in misps)
        range_ok = max(misps) - min(misps) < 0.03
        stable = "✓" if signs_same and range_ok else "✗"
        print(f"{stable:>10}")

    # ==========================================================================
    # STEP 3: Portfolio optimization (YOUR EXACT FORMULATION)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    print("\nFormulation:")
    print("  max  Σ X_i * (P_model_i - P_i) + Σ Y_i * (P_i - P_model_i)")
    print("  s.t. Σ X_i * P_i + Σ Y_i * (1 - P_i) ≤ Budget")
    print("       Delta = 0   (Σ z_i * S_i = 0)")
    print("       Theta ≥ 0   (positive theta)")
    print("       Fees = 0    (using mid prices)")

    budget = 1000.0  # $1,000 budget as specified
    portfolio = optimize_simple_portfolio(
        model=model,
        market_prices=market_prices,
        budget=budget,
        min_theta=0.0,  # Theta ≥ 0
    )

    print(f"\nBudget: ${budget:,.0f}")
    print(f"Solver status: {portfolio.solver_status}")

    print("\n" + "-" * 70)
    print("OPTIMAL POSITIONS")
    print("-" * 70)
    print(f"{'DTE':>6} {'P_mkt':>8} {'P_model':>8} {'Edge':>8} {'YES':>10} {'NO':>10} {'Net':>10}")
    print("-" * 70)

    for pos in portfolio.positions:
        if abs(pos.buy_yes) > 0.01 or abs(pos.buy_no) > 0.01:
            print(
                f"{pos.maturity:>6.0f} {pos.market_price:>8.2%} {pos.model_price:>8.2%} "
                f"{pos.edge_per_contract:>+8.2%} {pos.buy_yes:>10.2f} "
                f"{pos.buy_no:>10.2f} {pos.net_exposure:>+10.2f}"
            )

    print("-" * 70)

    # Summary
    total_yes = sum(p.buy_yes for p in portfolio.positions)
    total_no = sum(p.buy_no for p in portfolio.positions)
    print(f"{'TOTAL':>6} {'':<8} {'':<8} {'':<8} {total_yes:>10.2f} {total_no:>10.2f}")

    # Risk metrics
    print("\n" + "-" * 70)
    print("PORTFOLIO METRICS")
    print("-" * 70)
    print(f"Total Cost:     ${portfolio.total_cost:>10.2f}")
    print(f"Total Edge:     ${portfolio.total_edge:>10.4f}")
    print(f"Delta exposure:  {portfolio.delta_exposure:>+10.6f}  (target: 0)")
    print(f"Theta exposure:  {portfolio.theta_exposure:>+10.6f}  (target: ≥0)")
    if portfolio.total_cost > 0:
        print(f"Expected return: {portfolio.total_edge / portfolio.total_cost:>10.2%}")

    # ==========================================================================
    # STEP 4: Trade recommendations (UNHEDGED analysis)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRADE RECOMMENDATIONS")
    print("=" * 70)

    # Analyze unhedged opportunities based on mispricings
    print("\n--- DIRECTIONAL TRADES (based on stable mispricings) ---")
    print("\nLargest mispricings:")
    misp_with_idx = [(i, mispricings[i], market_data[i]) for i in range(len(mispricings))]
    misp_sorted = sorted(misp_with_idx, key=lambda x: abs(x[1]), reverse=True)

    for idx, misp, data in misp_sorted[:5]:
        spread_cost = data.ask_price - data.bid_price
        edge_after_spread = abs(misp) - spread_cost
        direction = "BUY YES" if misp > 0 else "BUY NO"

        print(f"\n  {data.contract_id.upper()} ({data.maturity:.0f} DTE)")
        print(f"    Market: {data.market_price:.1%} | Theoretical: {theo_prices[idx]:.1%}")
        print(f"    Mispricing: {misp:+.2%}")
        print(f"    Spread: {spread_cost:.1%}")
        print(f"    Edge after spread: {edge_after_spread:+.2%}")
        if edge_after_spread > 0.005:
            print(f"    → TRADE: {direction} (edge > spread)")
        else:
            print("    → SKIP: Edge doesn't cover spread")

    # ==========================================================================
    # STEP 4b: SPREAD TRADES (relative value)
    # ==========================================================================
    print("\n\n--- SPREAD TRADES (relative value) ---")
    print("\nLooking for contracts where mispricing changes sign...")

    for i in range(1, len(mispricings)):
        if mispricings[i] * mispricings[i - 1] < 0:  # Sign change
            d1, d2 = market_data[i - 1], market_data[i]
            m1, m2 = mispricings[i - 1], mispricings[i]

            print(f"\n  SPREAD: {d1.contract_id.upper()} vs {d2.contract_id.upper()}")
            p1_theo, p1_mkt = theo_prices[i - 1], d1.market_price
            p2_theo, p2_mkt = theo_prices[i], d2.market_price
            print(f"    {d1.contract_id}: Theo={p1_theo:.1%}, Mkt={p1_mkt:.1%}, Misp={m1:+.2%}")
            print(f"    {d2.contract_id}: Theo={p2_theo:.1%}, Mkt={p2_mkt:.1%}, Misp={m2:+.2%}")

            # The trade: sell the rich one, buy the cheap one
            if m1 < 0:  # d1 is overpriced
                leg1 = f"BUY NO {d1.contract_id.upper()}"
                leg2 = f"BUY YES {d2.contract_id.upper()}"
            else:
                leg1 = f"BUY YES {d1.contract_id.upper()}"
                leg2 = f"BUY NO {d2.contract_id.upper()}"

            spread_edge = abs(m1) + abs(m2)
            total_spread_cost = (d1.ask_price - d1.bid_price) + (d2.ask_price - d2.bid_price)

            print(f"    Combined edge: {spread_edge:.2%}")
            print(f"    Total spread cost: {total_spread_cost:.2%}")
            print(f"    Net edge: {spread_edge - total_spread_cost:.2%}")
            print(f"    → TRADE: {leg1} + {leg2}")

    # ==========================================================================
    # STEP 4c: Best single trade
    # ==========================================================================
    print("\n\n--- BEST SINGLE TRADE ---")

    # Find the trade with best edge after spread
    best_trade = None
    best_edge = 0

    for idx, misp, data in misp_with_idx:
        spread = data.ask_price - data.bid_price
        edge = abs(misp) - spread
        if edge > best_edge:
            best_edge = edge
            best_trade = (idx, misp, data, edge)

    if best_trade:
        idx, misp, data, edge = best_trade
        direction = "BUY YES" if misp > 0 else "BUY NO"
        price = data.ask_price if misp > 0 else (1 - data.bid_price)

        print(f"\n  ★ {direction} {data.contract_id.upper()}")
        print(f"    Days to expiry: {data.maturity:.0f}")
        print(f"    Entry price: {price:.1%}")
        print(
            f"    Theoretical value: {theo_prices[idx]:.1%}"
            if misp > 0
            else f"    Theoretical value: {1 - theo_prices[idx]:.1%}"
        )
        print(f"    Net edge: {edge:.2%}")
        print(f"    For $1000 position: ~${1000 * edge:.0f} expected profit")

    # ==========================================================================
    # STEP 5: Key insights
    # ==========================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Analyze implied hazard rates
    print("\n1. IMPLIED HAZARD RATE ANALYSIS:")
    print("   The bucket hazard rates show how the market prices event risk over time:")
    for i in range(len(bucket_hazards)):
        if i == 0:
            period = f"0-{maturities[i]:.0f} days"
        else:
            period = f"{maturities[i - 1]:.0f}-{maturities[i]:.0f} days"
        annualized = bucket_hazards[i] * 365
        print(f"   {period}: λ = {bucket_hazards[i]:.4f}/day ({annualized:.1%} annualized)")

    # Identify curve anomalies
    print("\n2. CURVE ANOMALIES:")
    for i in range(1, len(mispricings)):
        if mispricings[i] * mispricings[i - 1] < 0:  # Sign change
            c1, c2 = market_data[i - 1].contract_id, market_data[i].contract_id
            print(f"   Sign flip between {c1} and {c2}")
            print("   This suggests a potential relative value trade (spread)")

    # Check for calendar arbitrage
    print("\n3. CALENDAR ARBITRAGE CHECK:")
    price_diffs = np.diff(market_prices)
    if np.all(price_diffs >= -0.001):
        print("   ✓ No obvious calendar arbitrage (prices monotonically increasing)")
    else:
        for i in range(len(price_diffs)):
            if price_diffs[i] < -0.001:
                c1, p1 = market_data[i].contract_id, market_prices[i]
                c2, p2 = market_data[i + 1].contract_id, market_prices[i + 1]
                print(f"   ⚠ Potential arbitrage: {c1} ({p1:.1%}) > {c2} ({p2:.1%})")

    print("\n" + "=" * 70)
    print("DISCLAIMER: This is for educational purposes only. Past performance")
    print("does not guarantee future results. Trade at your own risk.")
    print("=" * 70)


if __name__ == "__main__":
    main()
