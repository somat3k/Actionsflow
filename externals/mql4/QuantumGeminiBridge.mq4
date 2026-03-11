#property strict

input string BridgeUrl = "http://127.0.0.1:8001/signal";
input string BridgeToken = "";
input int HistoryBars = 120;
input int RequestTimeoutMs = 3000;

datetime last_bar_time = 0;

int OnInit()
{
    return INIT_SUCCEEDED;
}

void OnTick()
{
    datetime current_bar = Time[0];
    if (current_bar == last_bar_time)
        return;
    last_bar_time = current_bar;

    string payload = BuildPayload();
    string response;
    if (!PostJson(BridgeUrl, payload, response))
    {
        Print("Bridge request failed.");
        return;
    }

    int signal = (int)ExtractJsonNumber(response, "\"signal\"");
    double confidence = ExtractJsonNumber(response, "\"confidence\"");
    Print("Bridge signal=", signal, " confidence=", DoubleToString(confidence, 2));
}

string BuildPayload()
{
    string candles_json = BuildCandlesJson();
    int long_count = 0;
    int short_count = 0;
    GetOpenPositionCounts(long_count, short_count);
    string symbol = JsonEscape(Symbol());
    string timeframe = JsonEscape(TimeframeToString(Period()));
    return StringFormat(
        "{\"platform\":\"mql4\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"candles\":%s,"
        "\"account\":{\"equity\":%G,\"balance\":%G,\"leverage\":%d},"
        "\"positions\":{\"long\":%d,\"short\":%d}}",
        symbol,
        timeframe,
        candles_json,
        AccountEquity(),
        AccountBalance(),
        AccountLeverage(),
        long_count,
        short_count
    );
}

string BuildCandlesJson()
{
    int max_bars = MathMin(HistoryBars, Bars);
    string body = "[";
    for (int i = max_bars - 1; i >= 0; i--)
    {
        string item = StringFormat(
            "{\"time\":\"%s\",\"open\":%G,\"high\":%G,\"low\":%G,\"close\":%G,\"volume\":%d}",
            FormatTimestamp(Time[i]),
            Open[i],
            High[i],
            Low[i],
            Close[i],
            (int)Volume[i]
        );
        body += item;
        if (i > 0)
            body += ",";
    }
    body += "]";
    return body;
}

bool PostJson(string url, string payload, string &response)
{
    char post[];
    StringToCharArray(payload, post, 0, WHOLE_ARRAY, CP_UTF8);
    char result[];
    string headers = "Content-Type: application/json\r\n";
    if (BridgeToken != "")
        headers += "X-Bridge-Token: " + BridgeToken + "\r\n";
    string result_headers;
    ResetLastError();
    int res = WebRequest("POST", url, headers, RequestTimeoutMs, post, result, result_headers);
    if (res == -1)
    {
        Print("WebRequest error: ", GetLastError());
        return false;
    }
    response = CharArrayToString(result, 0, -1, CP_UTF8);
    if (res < 200 || res >= 300)
    {
        PrintFormat("WebRequest HTTP error: status=%d", res);
        Print("Response headers: ", result_headers);
        Print("Response body: ", response);
        return false;
    }
    return true;
}

double ExtractJsonNumber(string json, string key)
{
    int key_pos = StringFind(json, key);
    if (key_pos < 0)
        return 0.0;
    int colon = StringFind(json, ":", key_pos);
    if (colon < 0)
        return 0.0;
    int end = StringFind(json, ",", colon);
    if (end < 0)
        end = StringFind(json, "}", colon);
    if (end < 0)
        end = StringLen(json);
    string raw = StringSubstr(json, colon + 1, end - colon - 1);
    raw = TrimString(raw);
    return StrToDouble(raw);
}

string TrimString(string value)
{
    StringTrimLeft(value);
    StringTrimRight(value);
    return value;
}

string TimeframeToString(int timeframe)
{
    if (timeframe == PERIOD_M1) return "M1";
    if (timeframe == PERIOD_M5) return "M5";
    if (timeframe == PERIOD_M15) return "M15";
    if (timeframe == PERIOD_M30) return "M30";
    if (timeframe == PERIOD_H1) return "H1";
    if (timeframe == PERIOD_H4) return "H4";
    if (timeframe == PERIOD_D1) return "D1";
    if (timeframe == PERIOD_W1) return "W1";
    if (timeframe == PERIOD_MN1) return "MN1";
    return "M" + IntegerToString(timeframe);
}

string FormatTimestamp(datetime value)
{
    string stamp = TimeToString(value, TIME_DATE | TIME_MINUTES | TIME_SECONDS);
    StringReplace(stamp, ".", "-");
    StringReplace(stamp, " ", "T");
    return stamp + "Z";
}

string JsonEscape(string value)
{
    string escaped = value;
    string backspace = CharToString(8);
    string formfeed = CharToString(12);
    StringReplace(escaped, "\\", "\\\\");
    StringReplace(escaped, "\"", "\\\"");
    StringReplace(escaped, "\r", "\\r");
    StringReplace(escaped, "\n", "\\n");
    StringReplace(escaped, "\t", "\\t");
    StringReplace(escaped, backspace, "\\b");
    StringReplace(escaped, formfeed, "\\f");
    return escaped;
}

void GetOpenPositionCounts(int &long_count, int &short_count)
{
    long_count = 0;
    short_count = 0;
    int total = OrdersTotal();
    for (int i = 0; i < total; i++)
    {
        if (!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
            continue;
        if (OrderSymbol() != Symbol())
            continue;
        if (OrderType() == OP_BUY)
            long_count++;
        else if (OrderType() == OP_SELL)
            short_count++;
    }
}
