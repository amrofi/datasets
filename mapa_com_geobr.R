# mapas com geobr e ggplot
library(geobr)
library(ggplot2)
library(sf)
library(dplyr)
library(rio)
library(readr)

options(timeout= 4000000)
metadata<-download_metadata() # para ver codigos
head(metadata)
all_mun_ms <- read_municipality(code_muni=50, year=2010)
plot(all_mun_ms)
class(all_mun_ms)

# Puxando os dados a serem plotados
# puxarei do github - Atlas Brasil do PNUD
library(readxl)
library(httr)
library(xlsx)

url <- "https://github.com/amrofi/atlasg/raw/master/dadosbrutos/atlas2013_dadosbrutos_pt.xlsx"
destfile <- "atlas2013_dadosbrutos_pt.xlsx"
#options(timeout= 4000000) # as vezes o site demora conforme sua internet
curl::curl_download(url, destfile)
atlas2013_dadosbrutos_pt <- read_excel(destfile, 
                                       sheet = "MUN 91-00-10")
dados_ms <- subset(atlas2013_dadosbrutos_pt, UF == '50' & ANO == '2010',
                   select=c(Codmun7,IDHM,IDHM_E,IDHM_L,IDHM_R))
View(dados_ms)

# juntar datasets all_mun_ms e dados_ms pelos descritores code_muni = Codmun7
dataset_final = left_join(all_mun_ms, dados_ms, by=c("code_muni"="Codmun7"))
plot(dataset_final)
max(dataset_final$IDHM) # 0.784
min(dataset_final$IDHM) # 0.526
ggplot() +
  geom_sf(data=dataset_final, aes(fill=IDHM), color= NA, size=.15)+
  labs(title="IDHM 2013 (ano base 2010) dos Municipíos de MS",
       caption='Fonte: Elaboração própria', size=8)+
  scale_fill_distiller(palette = "Greens", limits=c(0.5, 0.8),
                       name="IDHM")+
  theme_minimal()

# melhorando o mapa
library(ggplot2);library(ggspatial)
ggplot() +
  geom_sf(data=dataset_final, aes(fill=IDHM), color= "Black", size=.15)+
  labs(title="IDHM 2013 (ano base 2010) dos Municípios de MS",
       caption='Fonte: Elaboração própria', size=8)+
  scale_fill_distiller(palette = "RdGy", limits=c(0.5, 0.8),
                       name="IDHM")+
  theme_minimal() + 
  annotation_north_arrow(location = "bl", 
        which_north = "true", pad_x = unit(0.65, "in"), pad_y = unit(0.3, "in"), 
        style = north_arrow_fancy_orienteering) + 
  annotation_scale(location = "bl", width_hint = 0.3)

# Recomendação: situação em que quero separar unidades
# Vou gerar um excel com os dados de code_muni
library(writexl)
writexl::write_xlsx(as.data.frame(dataset_final[,1:2]),"dataset.xlsx")
# Vou criar uma coluna de dados no excel gerado e trazer para o mapa
# vou juntar ao dataset_final contendo code_muni
# como está com certeza na mesma ordem, posso apenas adicionar
library(readxl)
dataset <- read_excel("dataset.xlsx")
dataset_final2 = left_join(dataset_final,dataset, by=c("code_muni"="code_muni"))
ggplot() +
  geom_sf(data=dataset_final2, aes(fill=selecao), color= NA, size=.15)+
  labs(title="Selecao dos Municípios de MS",
       caption='Fonte: Elaboração própria', size=8)+
  theme_minimal() + 
  annotation_north_arrow(location = "bl", 
       which_north = "true", pad_x = unit(0.65, "in"), pad_y = unit(0.3, "in"), 
       style = north_arrow_fancy_orienteering) + 
  annotation_scale(location = "bl", width_hint = 0.3)

ggplot() +
  geom_sf(data=dataset_final2, aes(fill=factor(selecao2)), color= "Black", size=.15)+
  labs(title="Clusters dos Municipíos de MS",
       caption='Fonte: Elaboração própria', size=8)+
  theme_minimal() +
  annotation_north_arrow(location = "bl", 
       which_north = "true", pad_x = unit(0.65, "in"), pad_y = unit(0.3, "in"), 
       style = north_arrow_fancy_orienteering) + 
  annotation_scale(location = "bl", width_hint = 0.3)
