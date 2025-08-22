import pandas_datareader.data as web
import datetime

def fetch_macro_data():
    """
    Fetch US 10-Year Treasury Yield (DGS10) and CPI inflation (CPIAUCSL) from FRED
    """
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime.today()

    # 10-Year Treasury Yield
    yield_data = web.DataReader("DGS10", "fred", start, end)

    # CPI (Consumer Price Index)
    cpi_data = web.DataReader("CPIAUCSL", "fred", start, end)

    # Combine into one DataFrame
    data = yield_data.join(cpi_data)
    data.columns = ["10Y_Yield", "CPI"]

    # Reset index so Date is a column
    data.reset_index(inplace=True)

    data.to_csv("data_pipeline/macro_data.csv", index=False)

    print("✅ Saved macroeconomic data to data_pipeline/macro_data.csv")
    print(data.tail())


if __name__ == "__main__":
    fetch_macro_data()

