"""Calibration-only bounds for real attention and full-core query domains."""

def measurements(probes):
    for probe in probes:
        common={'family':probe['family'],'projection':probe['projection'],'seed':probe['seed']}
        for mode in probe['core']:
            for row in mode['results']:
                for domain,value in {'all':row['all'],**row['domains']}.items():
                    yield {**common,'stage':'core','mode':mode['mode'],'domain':domain,
                           'sample':row['name'],'metrics':value}
        for capture in probe['attention']:
            for row in capture['results']:
                values={'all':row['fp32'],**{k:v['fp32'] for k,v in row.get('query_domains',{}).items()}}
                for domain,value in values.items():
                    yield {**common,'stage':'attention','mode':row['mode'],'domain':domain,
                           'sample':f"step-{capture['step']}-block-{capture['block']}-layout-{row['head_major']}",
                           'metrics':value,'port_pass':row['port_pass']}

def freeze(probes):
    limits={}
    for row in measurements(probes):
        if row['seed'] not in (1001,1002):raise ValueError('non-calibration probe')
        if not row['metrics']['finite'] or not row.get('port_pass',True):raise ValueError('invalid calibration probe')
        gate=limits.setdefault(row['stage'],{}).setdefault(row['mode'],{}).setdefault(row['projection'],{}).setdefault(row['domain'],{})
        for metric in ('relative_l2','max_abs'):
            gate[metric]=max(gate.get(metric,0),row['metrics'][metric]*1.25+1e-7)
    return limits

def evaluate(probes,limits):
    results=[]
    for row in measurements(probes):
        gate=limits[row['stage']][row['mode']][row['projection']][row['domain']]
        numeric=row['metrics']['finite'] and all(row['metrics'][k]<=v for k,v in gate.items())
        results.append({**row,'numeric_pass':numeric,'limits':gate})
    return results
