CUDA Marketplace
===================

Datos del proyecto
------------------

- Objetivo:
  - Iniciarnos en la programación con CUDA
- Descripción del proyecto:
  - En base a una serie de productos con múltiples opciones de compra, determinar cuál es la mejor opción de compra para cada uno de estos productos.
El criterio de "mejor opción de compra" únicamente se basa en el precio de ésta.
- Autores:
  - [Adrià Jorquera Codina](https://github.com/adriajorquera "GitHub Adrià")
  - [Javier Ferrer González](https://github.com/JavierCane "GitHub Javier")
- Asignatura:
  - [Tarjetas Gráficas y Aceleradores (TGA)](http://www.fib.upc.edu/es/estudiar-enginyeria-informatica/assignatures/TGA.html)
- Fecha de entrega:
  - 23 de junio de 2015
- Curso:
  - 2014-2015
- Facultad:
  - [Facultad de Informática de Barcelona (FIB)](http://www.fib.upc.edu/)
- Universidad:
  - [Universidad Politecnica de Cataluña (UPC)](http://www.upc.edu/)

----------

Introducción
-------------

### Dominio
Para este ejercicio hemos supuesto que somos los gestores de un portal de compras on-line que recientemente ha recibido un incremento notable en el volumen de peticiones a sus servidores. Debido a este incremento, el rendimiento de la plataforma se ha visto afectado y debemos optimizar algunas partes de la aplicación para poder seguir ofreciendo una navegación fluida por nuestra web.

Nuestro portal de ventas es de tipo Marketplace. Con lo cual, integramos los catálogos que recibimos por parte de las diversas tiendas afiliadas. Cabe destacar el hecho de que tenemos un gran porcentaje de solapamiento entre los catálogos de distintas tiendas, produciendo así que un mismo producto tenga múltiples opciones de compra, cada una con su precio asociado.

El usuario final es el que selecciona con qué afiliado quiere comprar cada uno de los productos. Es así cómo se configura su carrito de compra para poder realizar un pedido. Dado que el usuario no tiene por qué ser consciente de cómo funciona un Marketplace, puede darse el caso de que tenga productos en su carrito de compra con afiliados que no tienen la opción de compra más barata.

Debido a este tipo de escenarios, tenemos un proceso en nuestro sistema que se encarga de asegurarle al usuario que en todo momento tiene el mejor carrito de compra posible y, si no es así, le damos la opción de mejorarlo aplicando las opciones de compra más baratas.

Nos hemos dado cuenta de que, uno de los puntos que consume más recursos de los servidores, es el propio cálculo de esta mejor opción de compra teniendo en cuenta todos los productos del carrito. Con lo cuál, nos hemos decidido a migrar este pequeño algoritmo de C++ a CUDA con tal de comprobar si ganaríamos en tiempo y podríamos así mantener una carga menor en los servidores.

Por darle un poco de volumen y poder tener así un estudio de tiempos que no sean despreciables, hemos asumido que el número de productos que debemos manejar se mueven por el orden de 30.000 productos. Por otra parte, debido al solapamiento de catálogo que comentábamos antes, cada producto tiene alrededor de 1.000 ofertas.

Nota: Esta idea está inspirada en una [funcionalidad real (Smart Cart) llevada a cabo por el marketplace Uvinum](http://blog.uvinum.es/te-ayudamos-uvinum-ahorrar-tus-comprasr-atento-nuestros-consejos-2372700 "Descripción del Smart Cart de Uvinum"). Que a pesar de tener un contexto donde el rendimiento no es tan crítico, ni tendría sentido llevar a cabo esta optimización en CUDA, sí que nos pareció interesante intentar adoptar un matiz mínimamente realista para la práctica.

### Estructuras de datos

Solución secuencial en C++
-------------

### Datos
- Implementación:
  - [main.cpp](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/main.cpp)
- Salida:
  - [main_cpp_output.txt](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/main_cpp_output.txt)

### Descripción

### Detalle de implementación
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

### Datos
- Implementación:
  - [main.cu](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/main.cu)
- Salida:
  - [main_cu_output.txt](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/main_cu_output.txt)

### Descripción

### Detalle de implementación
```
    tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = total_buy_options[global_thread_buy_option + STORE_ID_OFFSET];
    tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = total_buy_options[global_thread_buy_option + PRICE_OFFSET];
    __syncthreads();
    
    for (unsigned int stride = 2; stride <= blockDim.x; stride *= 2)
    {
        if (thread_id % stride == 0)
        {
            unsigned int next_buy_option_position = shared_thread_buy_option + stride;
            
	        // Si la opción de compra siguiente (teniendo en cuenta el stride) es mejor que la actual, la guardamos en el array de memoria compartida.
        }
        __syncthreads();
    }
```

Kernel CUDA con warps optimizados
-------------

### Datos
- Implementación:
  - [mainWarpsOptimized.cu](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/mainWarpsOptimized.cu)
- Salida:
  - [mainWarpsOptimized_output.txt](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/mainWarpsOptimized_output.txt)

### Descripción
En esta optimización, organizamos el trabajo que hace cada thread para tener un mejor acceso de memoria y un uso de WARPS más eficiente. Anteriormente dentro de un bloque los threads trabajaban primero los pares, luego los múltiplos de cuatro, seguido de los múltiplos de 8 y así sucesivamente. Como los threads se lanzan en WARPS, grupos de 32, se provocaba que los WARPS se vaciaran enseguida y se lanzaban 32 threads de los cuales pocos hacian trabajo útil.
Para solucionar el problema, se organiza que threads trabajan y sobre que elementos. La manera de conseguirlo es haciendo que la carga de trabajo recaiga sobre threads consecutivos. Así se logra que los WARPS cuando se lanzan todos sus threads tienen una cantidad de trabajo parecida.

### Detalle de implementación
```
    tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = total_buy_options[global_thread_buy_option + STORE_ID_OFFSET];
    tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = total_buy_options[global_thread_buy_option + PRICE_OFFSET];
    __syncthreads();
    
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

Kernel CUDA con más trabajo por thread
-------------

### Datos
- Implementación:
  - [mainWorkPerThreadOptimized.cu](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/mainWorkPerThreadOptimized.cu)
- Salida:
  - [mainWorkPerThreadOptimized_output.txt](https://github.com/JavierCane/Cuda-Marketplace-Knapsack/blob/master/mainWorkPerThreadOptimized_output.txt)

### Descripción

### Detalle de implementación
```
    if (first_price < second_price)
    {
        tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = first_store_id;
        tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = first_price;
    }

    else
    {
        tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = second_store_id;
        tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = second_price;
    }

    __syncthreads();
    
	// Bucle idéntico a la optimización anterior
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
| Más trabajo por thread| 4.697472	| 52.318 GB/s |
