CUDA Marketplace Knapsack
===================

Datos del proyecto
------------------

Objetivo
: Iniciarnos en la programación con CUDA

Descripción del proyecto
: En base a una serie de productos con múltiples opciones de compra, determinar cuál es la mejor opción de compra para cada uno de estos productos.
Para determinar la mejor oferta únicamente se tiene en cuenta el precio de éstas.

Autores
: [Adrià Jorquera Codina](https://github.com/adriajorquera "GitHub Adrià")
: [Javier Ferrer González](https://github.com/JavierCane "GitHub Javier")

Asignatura
: [Tarjetas Gráficas y Aceleradores (TGA)](http://www.fib.upc.edu/es/estudiar-enginyeria-informatica/assignatures/TGA.html)

Fecha de entrega
: 23 de junio de 2015

Curso
: 2014-2015

Facultad
: [Facultad de Informática de Barcelona (FIB)](http://www.fib.upc.edu/)

Universidad
: [Universidad Politecnica de Cataluña (UPC)](http://www.upc.edu/)

----------

Introducción
-------------

### Estructuras de datos

Solución secuencial en C++
-------------

### Implementación
```
    for(int product_iteration = 0; product_iteration < NUM_PRODUCTS; ++product_iteration)
    {
        for(int product_to_compare = ELEMENTS_PER_BUY_OPTION; product_to_compare < NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION; product_to_compare += ELEMENTS_PER_BUY_OPTION)
        {
            // Si la opción de compra actual es mejor que la que tenemos guardada de forma temporal, la substituimos para guardárnosla como mejor opción de compra.
        }
    }
```

Primera versión del kernel en CUDA
-------------

### Implementación
```
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            unsigned int next_buy_option_position = shared_thread_buy_option + stride * ELEMENTS_PER_BUY_OPTION;
            
	        // Si la opción de venta siguiente (teniendo en cuenta stride) es mejor que la actual, la guardamos en el array de memoria compartida.
        }
        __syncthreads();
    }
```

Kernel CUDA con warps optimizados
-------------

### Implementación
```
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            unsigned int next_buy_option_position = shared_thread_buy_option + stride * ELEMENTS_PER_BUY_OPTION;
            
	        // Si la opción de venta siguiente (teniendo en cuenta stride) es mejor que la actual, la guardamos en el array de memoria compartida.
        }
        __syncthreads();
    }
```

Conclusiones
-------------
### Rendimiento

Datos de la prueba:

- Número de opciones de compra por cada producto: 1024
- Número de productos (bloques del kernel): 30000
- Número de hilos por cada bloque: 1024
- Tamaño total del vector con todos los productos y todas las opciones de compra: 30720000

| Implementación     	| Tiempo total (en milisegs.) | Ancho de banda (en GB/s) |
|:----------------------|----------:|------------:|
| Secuencial en C++		|-----------|-------------|
| Kernel CUDA			| 12.416704	| 19.793 GB/s |
| Warps optimizados		| 8.137472	| 30.201 GB/s |



