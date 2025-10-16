# Título: Script para la Imputación de Datos de Calidad del Aire por Estación
# Descripción: Este código carga datos de calidad del aire de Bogotá,
#             realiza una imputación de valores faltantes para cada estación
#             utilizando el método k-Nearest Neighbors (k-NN), y luego generamos o exporta
#             el conjunto de datos completo.

# --- 1. Cargar Librerías ---
# Instalar paquetes si no están presentes
if (!require(readxl)) install.packages("readxl")
if (!require(dplyr)) install.packages("dplyr")
if (!require(VIM)) install.packages("VIM")
if (!require(writexl)) install.packages("writexl")

library(readxl)
library(dplyr)
library(VIM)
library(writexl)

# --- 2. Cargar el Archivo de Datos el archivo XLSX esté en tu directorio de trabajo o proporciona la ruta completa.
file_path <- "Contam-BOG-2021-2024-Localidades_1h.XLSX"
tryCatch({
  datos_originales <- read_excel(file_path)
}, error = function(e) {
  stop("Error al leer el archivo. Verifica que la ruta y el nombre del archivo sean correctos: ", e$message)
})

# --- 3. Preparación e Inspección de Datos ---
# Visualizar la estructura y las primeras filas
glimpse(datos_originales)
head(datos_originales)

# Resumen de valores faltantes por columna
print("Resumen de valores faltantes en el dataset original:")
colSums(is.na(datos_originales))

# --- 4. Proceso de Imputación por Estación ---
# comentario:Explicación del error: El error "subíndice fuera de los límites" a menudo
# ocurre en bucles cuando se intenta acceder a un índice que no existe.
# La advertencia sobre 'SO2' indica que esta columna está completamente vacía
# para al menos una estación. El enfoque robusto es iterar por cada estación,
# identificar las columnas que SÍ tienen datos para imputar en ese subconjunto,
# y luego realizar la imputación de manera segura.

# Obtener la lista de estaciones únicas
estaciones <- unique(datos_originales$Estacion)
if (length(estaciones) == 0) {
  stop("No se encontró la columna 'Estacion' o no contiene datos.")
}

# Crear una lista para almacenar los dataframes imputados de cada estación
lista_datos_imputados <- list()

# Bucle para procesar cada estación individualmente
for (estacion_actual in estaciones) {
  
  cat("\nProcesando estación:", estacion_actual, "\n")
  
  # Filtrar los datos para la estación actual
  datos_estacion <- datos_originales %>%
    filter(Estacion == estacion_actual)
  
  # Identificar columnas numéricas que se pueden imputar
  # (aquellas con algunos datos faltantes pero no TODOS faltantes)
  cols_a_imputar <- datos_estacion %>%
    select(where(is.numeric)) %>%
    select_if(function(col) sum(!is.na(col)) > 0) %>% 
    # Selecciona columnas que no estén completamente vacías
    names()
  
  if (length(cols_a_imputar) == 0) {
    cat("  -> No hay columnas con datos para imputar en esta estación. Saltando imputación.\n")
    # Añadir los datos de la estación sin cambios a la lista
    lista_datos_imputados[[estacion_actual]] <- datos_estacion
    next # Pasar a la siguiente estación
  }
  
  cat("  -> Columnas a imputar:", paste(cols_a_imputar, collapse = ", "), "\n")
  
  # Seleccionar solo las columnas numéricas para la imputación k-NN
  datos_para_imputar <- datos_estacion %>%
    select(all_of(cols_a_imputar))
  
  # Realizar la imputación k-NN
  # k=5 es un valor común, puedes ajustarlo si es necesario
  # imp_var = FALSE evita que se creen columnas adicionales indicando qué se imputó
  imputados_knn <- kNN(datos_para_imputar, k = 5, imp_var = FALSE)
  
  # Reemplazar las columnas originales con las columnas imputadas
  datos_estacion_imputados <- datos_estacion
  datos_estacion_imputados[, cols_a_imputar] <- imputados_knn
  
  # Guardar el dataframe imputado de la estación en la lista
  lista_datos_imputados[[estacion_actual]] <- datos_estacion_imputados
}

# --- 5. Consolidar y Exportar Resultados ---
# Combinar todos los dataframes de la lista en uno solo
datos_finales_imputados <- bind_rows(lista_datos_imputados)

# Verificar las dimensiones y los valores faltantes del resultado final
cat("\nDimensiones del dataset final imputado:", dim(datos_finales_imputados), "\n")
print("Resumen de valores faltantes en el dataset final:")
colSums(is.na(datos_finales_imputados))

# Exportar el dataframe final a un archivo .xlsx
output_file <- "datos_imputados.xlsx" #igual toco corregir el formato
tryCatch({
  write_xlsx(datos_finales_imputados, path = output_file)
  cat("\n¡Proceso completado! Los datos imputados se han guardado en:", output_file, "\n")
}, error = function(e) {
  cat("\nError al guardar el archivo. Asegúrate de tener permisos de escritura en el directorio.\n")
})
