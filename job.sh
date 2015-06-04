#!/bin/bash
### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N Marketplace-Knapsack
# Cambiar el shell
#$ -S /bin/bash

./Marketplace-Knapsack.exe
