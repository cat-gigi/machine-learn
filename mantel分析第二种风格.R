rm(list=ls())#好习惯，确保有干净的 R 环境
# setwd("C:/Users/Desktop/take")
library(linkET)
library(ggplot2)
library(ggtext)
library(dplyr)
library(RColorBrewer)
library(cols4all)
library(tidyverse)
library(linkET)
library(tidyverse)
library(RColorBrewer)

###################################此处需要修改路径##################################################
varespec <- read.csv("C:/Users/Xiaofeng Zhang/Desktop/mantel分析/1 原始数据/原始物种数据.csv", row.names = 1)  # 蝴蝶图矩阵，需要修改
varechem <- read.csv("C:/Users/Xiaofeng Zhang/Desktop/mantel分析/1 原始数据/原始环境因子数据.csv", row.names = 1) # 相关性分析矩阵，需要修改
###################################此处需要修改路径##################################################

#计算环境因子相关性系数：
cor2 <- correlate(varechem)
corr2 <- cor2 %>% as_md_tbl()
write.csv(corr2, file = "pearson_correlate(env&env).csv", row.names = TRUE)

head(corr2)
#mantel test:

###################################此处需要修改物种名和起始列##################################################
mantel <- mantel_test(varespec, varechem,
                      mantel_fun = 'mantel', #支持4种："mantel"使用vegan::mantel()；"mantel.randtest"使用ade4::mantel.randtest()；"mantel.rtest"使用ade4::mantel.rtest()；"mantel.partial"使用vegan::mantel.partial()
                      spec_select = list(Spec01 = 1:6,
                                         Spec02 = 8:18,
                                         Spec03 = 19:27,
                                         Spec05 = 28:37,
                                         Spec04 = 38:44
                      )) #这里分组为随机指定，具体实操需按自己的实际数据分组
###################################此处需要修改物种名和起始列##################################################



head(mantel)
write.csv(mantel, file = "mantel_result(bio&env).csv", row.names = TRUE)

#对mantel的r和P值重新赋值（设置绘图标签）：
mantel2 <- mantel %>%
  mutate(r = cut(r, breaks = c(-Inf, 0.25, 0.5, Inf),
                 labels = c("<0.25", "0.25-0.5", ">=0.5")),
         p = cut(p, breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                 labels = c("<0.001", "0.001-0.01", "0.01-0.05", ">= 0.05")))
head(mantel2)
#首先，绘制相关性热图(和上文相同):


##############################
p4 <- qcorrplot(cor2,
                grid_col = "#00468BFF",
                "white","#42B540FF",
                grid_size = 0.2,
                type = "upper",
                diag = FALSE) +
  geom_square() +
  scale_fill_gradientn(colours = c("#00468BFF",
                                   "white","#42B540FF"),
                       limits = c(-1, 1))

# 打印出来看看

#添加显著性标签：
p5 <- p4 +
  geom_mark(size = 4,
            only_mark = T,
            sig_level = c(0.05, 0.01, 0.001),
            sig_thres = 0.05,
            colour = 'white')
p5
#在相关性热图上添加mantel连线：
p6 <- p5 +
  geom_couple(data = mantel2,
              aes(colour = p, size = r),
              curvature = nice_curvature())
p6
#继续美化连线：
p7 <- p6 +
  scale_size_manual(values = c(1, 2, 3)) + #连线粗细
  scale_colour_manual(values = c4a('brewer.set2',4)) + #连线配色
  #修改图例：
  guides(size = guide_legend(title = "Mantel's r",
                             override.aes = list(colour = "grey35"),
                             order = 2),
         colour = guide_legend(title = "Mantel's p",
                               override.aes = list(size = 5),
                               order = 1),
         fill = guide_colorbar(title = "Pearson's r", order = 3))+
  theme(
    text = element_text(size = 16, family = "serif"),
    plot.title = element_text(size = 16, colour = "black", hjust = 0.5),
    legend.title = element_text(color = "black", size = 16),
    legend.text = element_text(color = "black", size = 16),
    axis.text.y = element_text(size = 16, color = "black", vjust = 0.5, hjust = 1, angle = 0),
    axis.text.x = element_text(size = 16, color = "black", vjust = 0.5, hjust = 0.5, angle = 0)
  )

p7
p7 <- p6 +
  scale_size_manual(values = c(1, 2, 3)) + #连线粗细
  scale_colour_manual(values = c4a('brewer.set2',4)) + #连线配色
  #修改图例：
  guides(size = guide_legend(title = "Mantel's r",
                             override.aes = list(colour = "grey35"),
                             order = 2),
         colour = guide_legend(title = "Mantel's p",
                               override.aes = list(size = 5),
                               order = 1),
         fill = guide_colorbar(title = "Pearson's r", order = 3))+
  theme(
    text = element_text(size = 16, family = "serif"),
    plot.title = element_text(size = 16, colour = "black", hjust = 0.5),
    legend.title = element_text(color = "black", size = 16),
    legend.text = element_text(color = "black", size = 16),
    axis.text.y = element_text(size = 16, color = "black", vjust = 0.5, hjust = 1, angle = 0),
    axis.text.x = element_text(size = 16, color = "black", vjust = 0.5, hjust = 0.5, angle = 0)
  )

p7

# 保存图片
ggsave("C:/Users/Xiaofeng Zhang/Desktop/mantel分析/mantel_plot.pdf", 
       p7, width = 10, height = 8, dpi = 300)

ggsave("C:/Users/Xiaofeng Zhang/Desktop/mantel分析/mantel_plot.tif", 
       p7, width = 10, height = 8, dpi = 300, device = "tiff")
