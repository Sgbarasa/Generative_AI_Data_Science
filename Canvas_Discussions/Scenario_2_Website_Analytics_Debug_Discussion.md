# Scenario 2: Website Analytics Debugging

## Problem and my prompting strategy

The marketing team needed a reliable explanation and correction for website engagement metrics, including bounce rates, session duration and device-level comparisons. I prompted AI as a senior product analytics engineer and included the defective function rather than just describing its symptoms. Most importantly, I specified that each input row represented a **page event**, not an entire session. That distinction guided the initial fix and subsequent tests. The demonstration data are synthetic because no original website logs were supplied.

## Initial AI prompt

> **Role:** Act as a senior product analytics engineer and Python debugging specialist. Diagnose the following function that returns impossible bounce rates and session times. Explain separately what is wrong with bounce rate, duration, per-session page views, and device aggregation; supply corrected pandas code. Assume multiple event rows per `session_id` and fields `session_id`, `page_views`, `duration`, `device_type`.

```python
def analyze_user_engagement(logs_df):
    metrics = {
        'bounce_rate': logs_df.groupby('session_id')['page_views'].apply(
            lambda x: x == 1).mean(),
        'avg_session_time': logs_df.groupby('session_id')['duration'].sum(),
        'pages_per_session': logs_df.groupby('session_id')['page_views'].mean()
    }
    device_metrics = logs_df.groupby('device_type').agg({
        'session_id': 'count', 'duration': 'mean', 'page_views': 'sum'
    })
    return metrics, device_metrics
```

## Initially generated code

```python
def draft_corrected_metrics(df):
    sessions = df.groupby('session_id').agg(
        views=('page_views', 'sum'),
        duration_seconds=('duration', 'sum'),
        device=('device_type', 'first')
    )
    bounce_rate = sessions.views.eq(1).mean() * 100
    avg_session_time = sessions.duration_seconds.mean()
    pages_per_session = sessions.views.mean()
    return bounce_rate, avg_session_time, pages_per_session
```

## Follow-up prompts and refinements

**Follow-up A:** “Explain specifically why the original logic is wrong, and distinguish symptoms demonstrated by the code from those that require investigating other pipeline stages. Add strict required-column checks, numeric casting, nonnegative duration and page-view validation, consistent device assignment per session, and a configurable policy for invalid rows (raise or quarantine with an audit count).”

**Follow-up B:** “Make `bounce_rate_pct` equal to the percentage of **sessions** with exactly one page view; compute average session time in seconds after aggregating each session; use distinct sessions per device, and add plots, assertions, and one synthetic invalid row to test error handling. Avoid silently manufacturing negative time corrections.”

## Final working solution

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

source_path = Path('website_logs.csv')
if source_path.exists():
    logs_df = pd.read_csv(source_path)
    print(f'Real file loaded: {source_path}')
else:
    logs_df = pd.DataFrame([
        ('S01',1,20,'mobile'),
        ('S02',1,80,'desktop'), ('S02',1,40,'desktop'),
        ('S03',1,0,'mobile'),
        ('S04',1,30,'mobile'), ('S04',1,20,'mobile'), ('S04',1,10,'mobile'),
        ('S05',1,10,'desktop'), ('S05',1,10,'desktop'),
        ('S06',1,-15,'mobile'),  # deliberately invalid row for demonstration
    ], columns=['session_id','page_views','duration','device_type'])
    print('Synthetic event-level test data loaded — not actual website logs')
print(logs_df.to_string(index=False))

def analyze_user_engagement(logs_df: pd.DataFrame, strict: bool = True):
    """Compute session-level engagement metrics from per-event page logs.

    Required: session_id, page_views (event increment), duration (seconds),
    device_type. strict=True raises on bad rows; strict=False quarantines them
    and returns an audit trail. A session must have only one device_type.
    """
    required = ['session_id','page_views','duration','device_type']
    missing = [x for x in required if x not in logs_df]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    work = logs_df[required].copy()
    work['session_id'] = work['session_id'].astype('string').str.strip()
    work['device_type'] = work['device_type'].astype('string').str.strip()
    work['page_views'] = pd.to_numeric(work['page_views'], errors='coerce')
    work['duration'] = pd.to_numeric(work['duration'], errors='coerce')
    valid = (
        work['session_id'].notna() & work['session_id'].ne('') &
        work['device_type'].notna() & work['device_type'].ne('') &
        work['page_views'].notna() & work['duration'].notna() &
        np.isfinite(work['page_views']) & np.isfinite(work['duration']) &
        work['page_views'].ge(0) & work['duration'].ge(0) &
        np.equal(work['page_views'], np.floor(work['page_views']))
    )
    audit = {
        'input_event_rows': int(len(work)),
        'invalid_event_rows': int((~valid).sum()),
    }
    if strict and audit['invalid_event_rows']:
        bad_indexes = work.index[~valid].tolist()
        raise ValueError(f'{len(bad_indexes)} invalid event row(s): {bad_indexes}')
    clean = work.loc[valid].copy()
    if clean.empty:
        raise ValueError('No valid event rows to aggregate')
    if clean.groupby('session_id')['device_type'].nunique().gt(1).any():
        raise ValueError('Some sessions have conflicting device assignments')

    # First aggregate all events to *one row per unique session*.
    sessions = (clean.groupby('session_id', as_index=False)
        .agg(page_views=('page_views','sum'),
             duration_seconds=('duration','sum'),
             device_type=('device_type','first')))
    sessions['bounce'] = sessions['page_views'].eq(1)
    total_sessions = len(sessions)
    metrics = {
        'sessions': int(total_sessions),
        'bounce_rate_pct': float(100 * sessions['bounce'].mean()),
        'avg_session_time_seconds': float(sessions['duration_seconds'].mean()),
        'pages_per_session': float(sessions['page_views'].mean()),
    }
    device_metrics = (sessions.groupby('device_type', as_index=False)
        .agg(sessions=('session_id','nunique'),
             bounce_rate_pct=('bounce', lambda col: 100 * col.mean()),
             avg_session_time_seconds=('duration_seconds','mean'),
             pages_per_session=('page_views','mean')))
    audit['valid_event_rows'] = int(len(clean))
    audit['valid_sessions'] = int(total_sessions)
    return metrics, sessions, device_metrics, audit

# Non-strict mode quarantines the one intentionally invalid event.
metrics, sessions, device_metrics, audit = analyze_user_engagement(logs_df, strict=False)
print('Summary:', metrics)
print('Data quality:', audit)
print('By device:')
print(device_metrics.to_string(index=False))

# Tests verify correct denominators, units, and error handling.
assert 0 <= metrics['bounce_rate_pct'] <= 100
assert (sessions['duration_seconds'] >= 0).all()
assert device_metrics['sessions'].sum() == metrics['sessions']
if not source_path.exists():
    assert metrics['sessions'] == 5
    assert metrics['bounce_rate_pct'] == 40.0
    assert metrics['avg_session_time_seconds'] == 44.0
    assert np.isclose(metrics['pages_per_session'], 1.8)
    assert audit['invalid_event_rows'] == 1
    try:
        analyze_user_engagement(logs_df, strict=True)
        raise AssertionError('Expected strict mode to flag negative duration')
    except ValueError:
        pass
print('Website engagement tests passed')

# Each visualization has one business question and explicit units.
ax = device_metrics.plot.bar(x='device_type', y='bounce_rate_pct', legend=False,
                             figsize=(8,4))
ax.set_title('Bounce rate by device (session-level)')
ax.set_xlabel('Device type')
ax.set_ylabel('Bounce rate (%)')
ax.set_ylim(0,100)
plt.xticks(rotation=0)
plt.tight_layout(); plt.show()

ax = device_metrics.plot.bar(x='device_type', y='avg_session_time_seconds',
                             legend=False, figsize=(8,4))
ax.set_title('Average session time by device')
ax.set_xlabel('Device type')
ax.set_ylabel('Seconds per session')
plt.xticks(rotation=0)
plt.tight_layout(); plt.show()

ax = device_metrics.plot.bar(x='device_type', y='pages_per_session',
                             legend=False, figsize=(8,4))
ax.set_title('Pages per session by device')
ax.set_xlabel('Device type')
ax.set_ylabel('Views per session')
plt.xticks(rotation=0)
plt.tight_layout(); plt.show()
```

## Results, validation and reflection

The root cause in the supplied function was an **aggregation-grain mismatch**. A bounce rate is the percentage of sessions with exactly one page view, not an average of event-level checks with the wrong denominator. Likewise, average session duration should be the mean of **per-session duration totals**, not a Series of totals; device reporting should count distinct sessions rather than event rows. One important nuance: the original Boolean mean, by itself, cannot exceed 100%; an above-100% bounce rate would imply an additional scaling or reporting error elsewhere in the pipeline.

My first refinement corrected session aggregation but still allowed malformed data. In the final code I added required-field and numeric checks, rejected negative duration, checked that a single session was assigned consistently to one device, and implemented **strict** error mode plus an alternative **quarantine** mode with an invalid-row audit count. The code treats duration as seconds **per event** and sums it for each session; a dataset with start/end timestamps or one row per session would require a different calculation.

On the synthetic event logs, the final output contained **5 valid sessions**, a **40.0% bounce rate**, **44.0 seconds average session time** and **1.8 pages per session**. One invalid event row was quarantined; the device summaries reconciled with the total session count. Assertions passed. The three charts compare bounce rate, average session duration and pages per session across devices.

The exercise showed that the best debugging prompt includes the actual failing code, an explicit data grain, units, expected mathematical definitions and failure modes. What would change in your approach if the source contained already aggregated sessions rather than individual page events?
