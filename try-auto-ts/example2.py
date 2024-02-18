from autots import AutoTS, load_live_daily, create_regressor

fred_key=None
gsa_key = None
forecast_length = 60

fred_series = [
    "DGS10",
    "T5YIE",
    "SP500",
    "DCOILWTICO",
    "DEXUSEU",
    "BAMLH0A0HYM2",
    "DAAA",
    "DEXUSUK",
    "T10Y2Y",
]

frequency = (
    "D"  # "infer" for automatic alignment, but specific offsets are most reliable
)

drop_most_recent=1

tickers = ["MSFT", "PG"]

trend_list = ["forecasting", "msft", "p&g"]

weather_event_types = ["%28Z%29+Winter+Weather", "%28Z%29+Winter+Storm"]

df = load_live_daily(
    long=False,
    fred_key=fred_key,
    fred_series=fred_series,
    tickers=tickers,
    trends_list=trend_list,
    earthquake_min_magnitude=5,
    weather_years=3,
    london_air_days=700,
    wikipedia_pages=['all', 'Microsoft', "Procter_%26_Gamble", "YouTube", "United_States"],
    gsa_key=gsa_key,
    gov_domain_list=None,  # ['usajobs.gov', 'usps.com', 'weather.gov'],
    gov_domain_limit=700,
    weather_event_types=weather_event_types,
    sleep_seconds=15,
)

regr_train, regr_fcst = create_regressor(
    df,
    forecast_length=forecast_length,
    frequency=frequency,
    drop_most_recent=drop_most_recent,
    scale=True,
    summarize="auto",
    backfill="bfill",
    fill_na="spline",
    holiday_countries={"US": None},  # requires holidays package
    encode_holiday_type=True,
)
