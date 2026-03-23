import torch
import numpy as np

# Tensor — это главный объект PyTorch.
# Как ndarray в NumPy, но с суперсилами (GPU + autograd).


# Создание тензора с помощью функции torch.tensor
# requires_grad=True — включаем автоградиент
# чтобы потом можно было посчитать градиенты"

# t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32 , requires_grad=True , device='cuda:0')   # Создаем тензор на GPU
# print(t)
# print(t.type())  # torch.FloatTensor
# print(t.dtype)   # torch.float32

# --------------------------------------------------------------

# tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32, requires_grad=True)


# print(tensor.type())         # torch.FloatTensor
# print(tensor.dtype)          # torch.float32
# print(tensor.shape)          # torch.Size([4, 2])
# print(tensor.size())         # torch.Size([4, 2])
# print(tensor.ndim)           # количество осьей
# print(tensor.device)         # cpu
# print(tensor[0,1,1].item())  # Получаем значение элемента в виде Python числа


# --------------------------------------------------------------

# zeros — это функция, которая создает тензор, заполненный нулями.

t_zeros = torch.zeros((2, 3), dtype=torch.float32)
# print(t_zeros)

# --------------------------------------------------------------

# torch.ones — это функция, которая создает тензор, заполненный единицами.

t_ones = torch.ones((2, 3), dtype=torch.int32)
# print(t_ones)

# --------------------------------------------------------------

# torch.rand — это функция, которая создает тензор,
# заполненный случайными числами из равномерного распределения на интервале [0, 1).

t_rand = torch.rand((2, 3), dtype=torch.float32)
# print(t_rand)

# --------------------------------------------------------------

# torch.arange — это функция, которая создает тензор, 
# заполненный числами в заданном диапазоне с определенным шагом.

t_arange = torch.arange(0, 10, step=2, dtype=torch.float32)
# print(t_arange)

# --------------------------------------------------------------

# torch.linspace — это функция, которая создает тензор, 
# заполненный числами, равномерно распределенными между начальным и конечным значениями.

t_linspace = torch.linspace(0, 10, steps=5, dtype=torch.float32)
# print(t_linspace)

# --------------------------------------------------------------

# torch.full_like — это функция, которая создает тензор,
# заполненный заданным значением, с той же формой и типом данных, что и другой тензор.

t_full_like = torch.full_like(t_ones, 8.0)
# print(t_full_like)

# --------------------------------------------------------------

# torch.eye — это функция, которая создает единичную матрицу 
# (матрицу с единицами на главной диагонали и нулями в остальных местах).

t_eye = torch.eye(3, dtype=torch.float32)
# print(t_eye)

# --------------------------------------------------------------

# torch.diag — это функция, которая создает диагональную матрицу из заданного вектора 
# или извлекает диагональные элементы из заданной матрицы.

t_diag = torch.diag(torch.tensor([1, 2, 3], dtype=torch.float32))
# print(t_diag)

# --------------------------------------------------------------

# torch.tril — это функция, которая возвращает нижнюю треугольную часть матрицы,
# обнуляя элементы выше главной диагонали.

t_tril = torch.tril(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32))
# print(t_tril)

# --------------------------------------------------------------

# torch.unspqueeze — это функция, которая добавляет новую ось в тензор, увеличивая его размерность.

tensor = torch.tensor([[1, 2], [3, 4]])
# print(tensor.shape)  # torch.Size([2, 2])
new_tensor = torch.unsqueeze(tensor, 0)  # Добавляем новую ось в начало
# print(new_tensor.shape)  # torch.Size([1, 2, 2])

# --------------------------------------------------------------

# torch.spqueeze это функция, которая удаляет ось из тензора, уменьшая его размерность.

tensor = torch.tensor([[1, 2], [3, 4]])


# --------------------------------------------------------------

# torch.view — это функция, которая изменяет форму тензора.


tensor = torch.tensor([1, 2, 3, 4, 5, 6], dtype=float)
tensor = tensor.view([2,3]) # Изменяем форму тензора на (2, 3)
tensor_mean = tensor.mean(dim=1)  # Вычисляем среднее значение по оси 1 (по строкам)
# print(tensor_mean)

# CPU & GPU
# print(torch.cpu.is_available())
# print(torch.cuda.is_available())

# --------------------------------------------------------------

# torch.to — это функция, которая изменяет устройство тензора.

tensor_cpu = torch.tensor([1, 2, 3, 4, 5, 6], dtype=float)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_gpu = tensor_cpu.to(device)

# print(tensor.device,tensor_gpu.device)
# tensor_cpu =  tensor_cpu.to(device)
# sum_tensor = tensor_cpu + tensor_gpu
# print(sum_tensor)

# --------------------------------------------------------------

# torch.from_numpy — это функция, которая создает тензор из NumPy массива.

arr = np.array([[1,2],[3,4]])
# print(type(arr))
t = torch.from_numpy(arr)
# print(t)
t_1 = torch.tensor(arr , dtype=torch.float32)
# print(t_1)

# --------------------------------------------------------------

# torch.save — это функция, которая сохраняет тензор в файле.

tensor = torch.tensor([1, 2, 3, 4, 5, 6], dtype=float)
torch.save(tensor, 'tensor.pt')

# --------------------------------------------------------------

# torch.load — это функция, которая загружает тензор из файла.

tensor = torch.load('tensor.pt')
# print(tensor)

# --------------------------------------------------------------

# torch.transpose — это функция, которая транспонирует тензор.

tensor = torch.tensor([[1, 2], [3, 4]])
# print(tensor)
# print(torch.transpose(tensor, 0, 1))

# --------------------------------------------------------------

# torch.permute — это функция, которая переставляет оси тензора.

tensor = torch.tensor([[1, 2], [3, 4]])
# print(tensor)
# print(torch.permute(tensor, (1, 0)))