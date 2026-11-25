# Scenario 2: Website Analytics Debugging

My prompt included the broken code because I wanted an explanation of the errors, not just a replacement function.

**Initial prompt:**
> Act as a Python analytics engineer. Fix a website metrics function built from event-level `session_id`, `page_views`, `duration` and `device_type` records. Explain the errors in bounce rate, session duration and pages per session, and return correct pandas calculations.

One issue in the original code was:

```python
'avg_session_time': logs_df.groupby('session_id')['duration'].sum(),
'pages_per_session': logs_df.groupby('session_id')['page_views'].mean()
```

The first line returned a series of session durations rather than their overall average. The second averaged page views at the wrong level. The first AI correction grouped rows into sessions, which was better, but did not handle bad data.

**Follow-up prompt:**
> Validate missing fields, negative durations, invalid page views and mismatched device types. Add a strict validation option, tests and charts showing engagement by device.

The final code calculates metrics from **sessions**, not events. On the demonstration data it returned **5 valid sessions, a 40% bounce rate and a 44-second average session time**. It also caught a deliberately negative duration. The charts compare bounce rate, time and pages per session across devices.

One detail worth checking: a Boolean mean cannot itself produce a percentage above 100%, so that reported symptom may involve another stage of the analytics pipeline. How would you calculate these metrics if your logs already had one record per session?
