"""Keep the requested engine option separate from observed worker layout."""


def observed_block_size(rows):
    if not rows:
        raise ValueError('Cannot determine block_size without worker observations')
    values = [row.get('capacity_layout', {}).get('block_tokens') for row in rows]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('Observed block_tokens must be a positive integer on every worker')
    if len(set(values)) != 1:
        raise ValueError('Workers report different block sizes; a single input alignment is unavailable')
    return values[0]


def input_block_size(plan):
    block = plan.get('observed_block_size')
    if block is None:
        block = plan['configuration']['server'].get('block_size')
    if type(block) is not int or block <= 0:
        raise ValueError('AISBench input preparation needs the actual block_size: use a managed '
                         'experiment, attach with --service-run, use --observe with BundleObserver, '
                         'or explicitly configure a known server.block_size. No default is assumed.')
    return block
