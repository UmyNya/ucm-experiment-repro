"""Deltas of independent cumulative native counters, preserving per-worker evidence."""


def summarize(before, after):
    previous={r['pid']:r for r in before}
    current={r['pid']:r for r in after}
    if previous.keys()!=current.keys() or not previous:
        raise ValueError('Native metric worker identities changed')
    workers=[];totals={};counts={};sums={}
    for pid in sorted(previous):
        a,b=previous[pid],current[pid]
        row={'pid':pid,'gauges_after':b.get('gauges',{})}
        for field,aggregate in [('counters',totals),('histogram_counts',counts),('histogram_sums',sums)]:
            old,new=a.get(field,{}),b.get(field,{})
            changes={name:new.get(name,0)-old.get(name,0) for name in old.keys()|new.keys()}
            if any(value<0 for value in changes.values()):
                raise ValueError('Cumulative native metric reset during measurement: '+field)
            row[field]=changes
            for name,value in changes.items():aggregate[name]=aggregate.get(name,0)+value
        workers.append(row)
    return {'passed':True,'workers':workers,'counter_totals':totals,'histogram_counts':counts,'histogram_sums':sums,
            'histogram_means':{name:sums.get(name,0)/count if count else None for name,count in counts.items()},
            'scope':'Independent cumulative native counters between physically drained boundaries; gauges remain per worker'}
