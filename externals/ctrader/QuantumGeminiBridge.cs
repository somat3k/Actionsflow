using cAlgo.API;
using cAlgo.API.Internals;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.Internet)]
    public class QuantumGeminiBridge : Robot
    {
        [Parameter("Bridge URL", DefaultValue = "http://127.0.0.1:8001/signal")]
        public string BridgeUrl { get; set; }

        [Parameter("History Bars", DefaultValue = 120)]
        public int HistoryBars { get; set; }

        private HttpClient _client;

        protected override void OnStart()
        {
            _client = new HttpClient();
        }

        protected override void OnBar()
        {
            var request = BuildRequest();
            var json = JsonSerializer.Serialize(request);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = _client.PostAsync(BridgeUrl, content).GetAwaiter().GetResult();
            var body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            if (!response.IsSuccessStatusCode)
            {
                Print("Bridge error: {0}", body);
                return;
            }

            var bridgeResponse = JsonSerializer.Deserialize<BridgeResponse>(body);
            if (bridgeResponse == null)
            {
                Print("Bridge response empty.");
                return;
            }

            Print("Signal={0} Confidence={1:F2} Regime={2}", bridgeResponse.Signal, bridgeResponse.Confidence, bridgeResponse.Regime);
        }

        protected override void OnStop()
        {
            _client?.Dispose();
        }

        private BridgeRequest BuildRequest()
        {
            var candles = new List<Candle>();
            var count = Math.Min(HistoryBars, Bars.Count);
            for (int i = count - 1; i >= 0; i--)
            {
                candles.Add(new Candle
                {
                    Time = Bars.OpenTimes[i].ToString("o"),
                    Open = Bars.OpenPrices[i],
                    High = Bars.HighPrices[i],
                    Low = Bars.LowPrices[i],
                    Close = Bars.ClosePrices[i],
                    Volume = Bars.TickVolumes[i]
                });
            }

            return new BridgeRequest
            {
                Platform = "ctrader",
                Symbol = SymbolName,
                Timeframe = TimeFrame.ToString(),
                Candles = candles,
                Account = new AccountSnapshot
                {
                    Equity = Account.Equity,
                    Balance = Account.Balance,
                    Leverage = Account.Leverage
                },
                Positions = new PositionSnapshot
                {
                    Long = Positions.FindAll(SymbolName, TradeType.Buy).Length,
                    Short = Positions.FindAll(SymbolName, TradeType.Sell).Length
                }
            };
        }
    }

    public class BridgeRequest
    {
        [JsonPropertyName("platform")]
        public string Platform { get; set; }

        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("timeframe")]
        public string Timeframe { get; set; }

        [JsonPropertyName("candles")]
        public List<Candle> Candles { get; set; }

        [JsonPropertyName("account")]
        public AccountSnapshot Account { get; set; }

        [JsonPropertyName("positions")]
        public PositionSnapshot Positions { get; set; }
    }

    public class Candle
    {
        [JsonPropertyName("time")]
        public string Time { get; set; }

        [JsonPropertyName("open")]
        public double Open { get; set; }

        [JsonPropertyName("high")]
        public double High { get; set; }

        [JsonPropertyName("low")]
        public double Low { get; set; }

        [JsonPropertyName("close")]
        public double Close { get; set; }

        [JsonPropertyName("volume")]
        public double Volume { get; set; }
    }

    public class AccountSnapshot
    {
        [JsonPropertyName("equity")]
        public double Equity { get; set; }

        [JsonPropertyName("balance")]
        public double Balance { get; set; }

        [JsonPropertyName("leverage")]
        public double Leverage { get; set; }
    }

    public class PositionSnapshot
    {
        [JsonPropertyName("long")]
        public double Long { get; set; }

        [JsonPropertyName("short")]
        public double Short { get; set; }
    }

    public class BridgeResponse
    {
        [JsonPropertyName("status")]
        public string Status { get; set; }

        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("signal")]
        public int Signal { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("regime")]
        public string Regime { get; set; }
    }
}
