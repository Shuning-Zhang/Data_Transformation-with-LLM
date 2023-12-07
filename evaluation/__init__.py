def task_acc(name:str, lower_bound:int, upper_bound: int):
    if name == 'foofah':
        from .foofah import calculate_acc
        return calculate_acc(lower_bound, upper_bound)
    else:
        raise NotImplementedError
    
print(task_acc('foofah', 2,12))