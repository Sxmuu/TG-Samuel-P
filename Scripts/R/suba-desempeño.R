# Título: Script para Evaluar la Imputación k-NN en Suba para PM2.5
# Descripción: Este código se enfoca en la estación "Suba", identificada como
#              robusta por el análisis de calidad de datos. Simula la pérdida
#              de datos para PM2.5 y evalúa la precisión del método k-NN.

# --- 1. Cargar Librerías ---
if (!require(readxl)) install.packages("readxl")
if (!require(dplyr)) install.packages("dplyr")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(VIM)) install.packages("VIM")
if (!require(janitor)) install.packages("janitor")

library(readxl)
library(dplyr)
library(ggplot2)
library(VIM)
library(janitor)

# --- 2. Cargar y Preprocesar Datos ---
file_path <- "Contam-BOG-2021-2024-Localidades_1h.XLSX"
tryCatch({
  datos_originales <- read_excel(file_path)
}, error = function(e) {
  stop("Error al leer el archivo. Verifica la ruta: ", e$message)
})

# --- SELECCIÓN DE LA ESTACIÓN RECOMENDADA ---
# "Suba", que tiene datos más completos.
estacion_seleccionada <- "Suba"

# Corregir formato de fecha y filtrar para la estación seleccionada
datos_estacion <- datos_originales %>%
  mutate(DateTime = excel_numeric_to_date(DateTime, tz = "America/Bogota")) %>%
  filter(Estacion == estacion_seleccionada)

cat(paste("Datos cargados y filtrados para la estación:", estacion_seleccionada, "\n\n"))

# --- 3. Preparar Simulación ---

# Función para el Índice de Concordancia de Willmott (d)
willmott_d <- function(observado, predicho) {
  valid_indices <- !is.na(observado) & !is.na(predicho)
  observado <- observado[valid_indices]
  predicho <- predicho[valid_indices]
  if (length(observado) < 2) return(NA)
  numerador <- sum((predicho - observado)^2)
  denominador <- sum((abs(predicho - mean(observado)) + abs(observado - mean(observado)))^2)
  if (denominador == 0) return(1)
  return(1 - (numerador / denominador))
}

# Parámetros y preparación
porcentajes_faltantes <- c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
contaminante_a_evaluar <- "PM25"
columnas_predictoras <- c("PM10", "NO2", "OZONO", "SO2", "CO", "Temperatura")

datos_completos_sim <- datos_estacion %>%
  select(all_of(contaminante_a_evaluar), all_of(columnas_predictoras)) %>%
  na.omit()

if (nrow(datos_completos_sim) < 50) {
  stop(paste("Datos completos insuficientes para la simulación en", estacion_seleccionada))
}

cat(paste("Número de registros completos para la simulación:", nrow(datos_completos_sim), "\n\n"))

# Dataframe para resultados
resultados_evaluacion <- data.frame()

# --- 4. Ejecutar la Simulación ---
cat(paste("Iniciando simulación para PM2.5 en", estacion_seleccionada, "...\n"))

for (porcentaje in porcentajes_faltantes) {
  n_total <- nrow(datos_completos_sim)
  n_faltantes <- floor(n_total * porcentaje)
  indices_na <- sample(1:n_total, size = n_faltantes)
  
  datos_simulados_con_na <- datos_completos_sim
  datos_simulados_con_na[indices_na, contaminante_a_evaluar] <- NA
  
  imputados_knn <- kNN(datos_simulados_con_na, k = 5, imp_var = FALSE)
  
  valores_reales <- datos_completos_sim[[contaminante_a_evaluar]][indices_na]
  valores_imputados <- imputados_knn[[contaminante_a_evaluar]][indices_na]
  
  if (length(valores_reales) > 1 && var(valores_reales, na.rm=TRUE) > 0) {
    resultados_temp <- data.frame(
      PorcentajeFaltante = porcentaje * 100,
      Correlacion_R = cor(valores_reales, valores_imputados, use = "complete.obs"),
      MAE = mean(abs(valores_reales - valores_imputados), na.rm = TRUE),
      Indice_d = willmott_d(valores_reales, valores_imputados)
    )
    resultados_evaluacion <- rbind(resultados_evaluacion, resultados_temp)
  }
}

# --- 5. Mostrar Resultados y Gráfico ---
cat(paste("\n--- Tabla de Desempeño para PM2.5 en", estacion_seleccionada, "---\n"))
print(resultados_evaluacion, row.names = FALSE)

if (nrow(resultados_evaluacion) > 0) {
  grafico_final <- ggplot(resultados_evaluacion, aes(x = PorcentajeFaltante, y = Correlacion_R)) +
    geom_line(color = "#0072B2", size = 1) +
    geom_point(color = "#0072B2", size = 3) +
    labs(
      title = paste("Desempeño de Imputación k-NN para PM2.5 en", estacion_seleccionada),
      x = "Porcentaje de Datos Faltantes (%)",
      y = "Coeficiente de Correlación (R)"
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal(base_size = 14)
  
  print(grafico_final)
  ggsave(paste0("desempeno_imputacion_", tolower(estacion_seleccionada), "_pm25.png"), grafico_final, width = 10, height = 7)
}
