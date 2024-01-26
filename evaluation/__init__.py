def task_acc(name:str, lower_bound:int, upper_bound: int, model: str = None):
    if name == 'foofah':
        from .foofah import calculate_acc
        return calculate_acc(lower_bound, upper_bound)
    elif name == 'llm':
        from .llm import calculate_acc
        return calculate_acc(model, lower_bound, upper_bound)
    else:
        raise NotImplementedError
    
print(task_acc('foofah', 2,12))