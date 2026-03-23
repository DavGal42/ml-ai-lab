import torch
import torch.nn as nn

# модель
model = nn.Sequential(
    nn.Linear(2,2),
    nn.ReLU(),
    nn.Linear(2,1)
)

x = torch.tensor([[2.0,3.0]])

# функция печати графа
def print_graph(grad_fn, indent=0):

    if grad_fn is None:
        return

    print(" " * indent + str(grad_fn))

    for next_fn, _ in grad_fn.next_functions:
        print_graph(next_fn, indent + 4)


# ---------------------------------
# WITHOUT torch.no_grad()
# ---------------------------------

print("\n===== WITHOUT torch.no_grad() =====")

y_pred = model(x)

print("\nPrediction:", y_pred)
print("\nComputation graph:\n")

print_graph(y_pred.grad_fn)



# ---------------------------------
# WITH torch.no_grad()
# ---------------------------------

print("\n\n===== WITH torch.no_grad() =====")

with torch.no_grad():
    y_pred2 = model(x)

print("\nPrediction:", y_pred2)

print("\nComputation graph:")

if y_pred2.grad_fn is None:
    print("No graph stored (gradients disabled)")
else:
    print_graph(y_pred2.grad_fn)