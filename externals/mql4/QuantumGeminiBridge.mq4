#property strict

input string BridgeUrl = "http://127.0.0.1:8001/signal";
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
    return StringFormat(
        "{\"platform\":\"mql4\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"candles\":%s}",
        Symbol(),
        TimeframeToString(Period()),
        candles_json
    );
}

string BuildCandlesJson()
{
    int max_bars = MathMin(HistoryBars, Bars - 1);
    string body = "[";
    for (int i = max_bars - 1; i >= 0; i--)
    {
        string item = StringFormat(
            "{\"time\":\"%s\",\"open\":%G,\"high\":%G,\"low\":%G,\"close\":%G,\"volume\":%d}",
            TimeToString(Time[i], TIME_DATE | TIME_MINUTES | TIME_SECONDS),
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
    string result_headers;
    ResetLastError();
    int res = WebRequest("POST", url, headers, RequestTimeoutMs, post, result, result_headers);
    if (res == -1)
    {
        Print("WebRequest error: ", GetLastError());
        return false;
    }
    response = CharArrayToString(result, 0, -1, CP_UTF8);
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
