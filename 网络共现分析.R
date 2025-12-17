library(igraph)
library(Hmisc)
library(RColorBrewer)

# 1. 读取数据
data <- read.csv("D:/5 共现网络分析/1 原始数据/ASV_ba_rep.csv", 
                 row.names = 1, check.names = FALSE)
print(paste("原始数据：", nrow(data), "样本 ×", ncol(data), "OTU"))

# 2. 过滤低丰度OTU
max_abundance <- apply(data, 2, max)
filtered_data <- data[, max_abundance >= 0.0001]
print(paste("过滤后：", nrow(filtered_data), "样本 ×", ncol(filtered_data), "OTU"))

# 3. 计算相关性
spearman_cor <- rcorr(as.matrix(filtered_data), type = "spearman")
cor_matrix <- spearman_cor$r
p_matrix <- spearman_cor$P

# 4. FDR校正
p_vector <- p_matrix[lower.tri(p_matrix)]
q_vector <- p.adjust(p_vector, method = "BH")
q_matrix <- p_matrix
q_matrix[lower.tri(q_matrix)] <- q_vector
q_matrix[upper.tri(q_matrix)] <- t(q_matrix)[upper.tri(q_matrix)]

# 5. 筛选显著相关
significant_mask <- (abs(cor_matrix) > 0.6) & (q_matrix < 0.05)
diag(significant_mask) <- FALSE

# 6. 提取显著OTU
otu_has_sig <- apply(significant_mask, 1, function(x) any(x))
sig_otu_names <- colnames(filtered_data)[otu_has_sig]
sig_data <- filtered_data[, sig_otu_names]
print(paste("显著OTU：", length(sig_otu_names), "个"))

# 7. 创建边列表
cor_subset <- cor_matrix[sig_otu_names, sig_otu_names]
q_subset <- q_matrix[sig_otu_names, sig_otu_names]
p_subset <- p_matrix[sig_otu_names, sig_otu_names]

sig_mask <- (abs(cor_subset) > 0.6) & (q_subset < 0.05)
diag(sig_mask) <- FALSE
edges <- which(sig_mask & upper.tri(sig_mask), arr.ind = TRUE)

if (nrow(edges) > 0) {
  # 创建边数据
  edge_list <- data.frame(
    from = rownames(cor_subset)[edges[, 1]],
    to = rownames(cor_subset)[edges[, 2]],
    weight = abs(cor_subset[edges]),
    correlation = cor_subset[edges],
    p_value = p_subset[edges],
    q_value = q_subset[edges]
  )
  
  # 8. 创建igraph网络
  g <- graph_from_data_frame(edge_list, directed = FALSE)
  
  # 9. 社区检测（模块识别）- 使用Louvain算法
  set.seed(123)  # 设置随机种子保证结果可重复
  community <- cluster_louvain(g)
  
  # 10. 为每个社区生成颜色（节点颜色）
  num_communities <- length(unique(membership(community)))
  
  # 生成节点颜色（同一模块颜色相同）
  if (num_communities <= 12) {
    node_colors <- brewer.pal(max(3, num_communities), "Set3")
  } else if (num_communities <= 20) {
    # 合并多个调色板
    node_colors <- colorRampPalette(brewer.pal(8, "Set2"))(num_communities)
  } else {
    # 使用更丰富的颜色
    node_colors <- colorRampPalette(brewer.pal(12, "Set3"))(num_communities)
  }
  
  # 11. 节点属性：基于社区的颜色和位置
  V(g)$community <- membership(community)
  V(g)$color <- node_colors[membership(community)]
  V(g)$degree <- degree(g)
  V(g)$betweenness <- betweenness(g)
  
  # 12. 边属性 - 只用红和绿两种颜色
  # 正相关：红色，负相关：绿色
  edge_colors <- ifelse(edge_list$correlation > 0, "red", "green")
  E(g)$color <- edge_colors
  E(g)$correlation_type <- ifelse(edge_list$correlation > 0, "positive", "negative")
  
  # 根据相关性强度设置边的透明度
  edge_alpha <- 0.4 + 0.4 * (abs(edge_list$correlation) - 0.6) / (1.0 - 0.6)
  edge_alpha <- pmin(pmax(edge_alpha, 0.4), 0.8)
  
  # 13. 创建边权重：同一社区的边权重更大（使它们更聚集）
  edge_weights <- rep(1, ecount(g))
  
  # 为每条边检查两个端点是否在同一社区
  for (i in 1:ecount(g)) {
    edge_ends <- ends(g, i)
    from_community <- V(g)$community[V(g)$name == edge_ends[1]]
    to_community <- V(g)$community[V(g)$name == edge_ends[2]]
    
    if (from_community == to_community) {
      edge_weights[i] <- 5  # 同一社区的边权重更大
    }
  }
  
  E(g)$weight <- edge_weights
  
  # 14. 创建基于社区的布局 - 确保同一模块节点聚集
  layout_community_clustered <- function(g, community) {
    comm_members <- membership(community)
    num_communities <- max(comm_members)
    
    # 为每个社区创建单独的布局
    community_layouts <- list()
    for (i in 1:num_communities) {
      nodes_in_community <- which(comm_members == i)
      if (length(nodes_in_community) > 0) {
        subgraph <- induced_subgraph(g, nodes_in_community)
        if (vcount(subgraph) == 1) {
          # 单个节点
          layout_matrix <- matrix(c(0, 0), nrow = 1, ncol = 2)
        } else {
          # 使用FR布局创建社区内部布局
          layout_matrix <- layout_with_fr(subgraph, niter = 1000)
        }
        community_layouts[[i]] <- layout_matrix
      }
    }
    
    # 将各社区布局组合到一起
    final_layout <- matrix(0, nrow = vcount(g), ncol = 2)
    
    # 将社区放在圆形上，确保模块之间分开
    radius <- 4.0  # 增加半径让社区更分散
    
    for (i in 1:num_communities) {
      nodes_in_community <- which(comm_members == i)
      if (length(nodes_in_community) > 0) {
        # 社区中心位置
        angle_center <- 2 * pi * (i - 1) / num_communities
        center_x <- radius * cos(angle_center)
        center_y <- radius * sin(angle_center)
        
        # 将社区内部布局平移到中心位置
        community_scale <- 0.6  # 缩小社区内部布局，使节点更紧密
        
        if (is.matrix(community_layouts[[i]])) {
          # 如果社区有多个节点，标准化布局
          if (nrow(community_layouts[[i]]) > 1) {
            # 标准化到单位圆内
            layout_scaled <- community_layouts[[i]]
            layout_range <- apply(layout_scaled, 2, function(x) diff(range(x)))
            if (all(layout_range > 0)) {
              layout_scaled[, 1] <- (layout_scaled[, 1] - mean(layout_scaled[, 1])) / layout_range[1]
              layout_scaled[, 2] <- (layout_scaled[, 2] - mean(layout_scaled[, 2])) / layout_range[2]
            }
            final_layout[nodes_in_community, 1] <- center_x + community_scale * layout_scaled[, 1]
            final_layout[nodes_in_community, 2] <- center_y + community_scale * layout_scaled[, 2]
          } else {
            # 单个节点
            final_layout[nodes_in_community, 1] <- center_x
            final_layout[nodes_in_community, 2] <- center_y
          }
        }
      }
    }
    
    return(final_layout)
  }
  
  # 使用社区聚集布局
  layout_selected <- layout_community_clustered(g, community)
  
  # 15. 可视化
  par(mar = c(0, 0, 2, 0))
  
  # 设置顶点大小（根据度来调整）
  vertex_size <- sqrt(V(g)$degree + 1) * 3
  
  # 创建带有透明度的边颜色
  edge_plot_colors <- character(ecount(g))
  for (i in 1:ecount(g)) {
    if (edge_list$correlation[i] > 0) {
      # 正相关：红色
      edge_plot_colors[i] <- adjustcolor("red", alpha.f = edge_alpha[i])
    } else {
      # 负相关：绿色
      edge_plot_colors[i] <- adjustcolor("green", alpha.f = edge_alpha[i])
    }
  }
  
  # 绘制网络
  plot(g, 
       layout = layout_selected,
       vertex.size = vertex_size,
       vertex.label = NA,  # 不显示标签
       vertex.frame.color = "white",
       vertex.color = V(g)$color,
       edge.width = 0.8 * sqrt(edge_list$weight),
       edge.color = edge_plot_colors,
       main = paste("Ecological Network with", num_communities, "Modules (N =", vcount(g), ")"))
  
  # 添加图例 - 节点颜色（模块）
  if (num_communities <= 15) {
    legend("topright", 
           legend = paste("Module", 1:num_communities, " (", as.vector(table(V(g)$community)), ")", sep=""),
           col = node_colors[1:num_communities],
           pch = 16,
           bty = "n",
           pt.cex = 1.5,
           cex = 0.7,
           ncol = ifelse(num_communities > 8, 2, 1))
  }
  
  # 添加图例 - 边颜色（红和绿）
  legend("bottomright",
         legend = c("Positive correlation (red)", "Negative correlation (green)"),
         col = c("red", "green"),
         lwd = 2,
         bty = "n",
         cex = 0.7)
  
  # 16. 保存网络数据（包含社区信息）
  # 添加布局信息到节点属性
  V(g)$layout_x <- layout_selected[, 1]
  V(g)$layout_y <- layout_selected[, 2]
  
  # 17. 保存为GraphML
  write_graph(g, "D:/5 共现网络分析/1 原始数据/network_community.graphml", format = "graphml")
  
  # 18. 保存社区分配信息
  community_info <- data.frame(
    OTU = V(g)$name,
    Community = V(g)$community,
    Node_Color = V(g)$color,
    Degree = V(g)$degree,
    Betweenness = V(g)$betweenness,
    Layout_X = V(g)$layout_x,
    Layout_Y = V(g)$layout_y
  )
  
  write.csv(community_info, 
            "D:/5 共现网络分析/1 原始数据/community_assignment.csv",
            row.names = FALSE)
  
  # 19. 保存边信息
  edge_info <- data.frame(
    From = edge_list$from,
    To = edge_list$to,
    Correlation = edge_list$correlation,
    Correlation_Abs = edge_list$weight,
    P_value = edge_list$p_value,
    Q_value = edge_list$q_value,
    Edge_Type = E(g)$correlation_type,
    Edge_Color = E(g)$color,
    Edge_Weight = E(g)$weight
  )
  
  write.csv(edge_info,
            "D:/5 共现网络分析/1 原始数据/edge_info.csv",
            row.names = FALSE)
  
  # 20. 保存网络统计信息
  network_stats <- data.frame(
    Metric = c("Total_nodes", "Total_edges", "Number_of_modules", 
               "Modularity", "Average_degree", "Average_clustering",
               "Graph_density", "Diameter", "Average_path_length"),
    Value = c(vcount(g), ecount(g), num_communities,
              modularity(community), mean(degree(g)), 
              transitivity(g, type = "average"),
              edge_density(g), diameter(g), 
              mean_distance(g, directed = FALSE))
  )
  
  write.csv(network_stats,
            "D:/5 共现网络分析/1 原始数据/network_statistics.csv",
            row.names = FALSE)
  
  # 21. 打印统计信息
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("NETWORK ANALYSIS SUMMARY\n")
  cat(rep("=", 60), "\n\n")
  cat(paste("• GraphML file saved: network_community.graphml\n"))
  cat(paste("• Community assignment saved: community_assignment.csv\n"))
  cat(paste("• Edge information saved: edge_info.csv\n"))
  cat(paste("• Network statistics saved: network_statistics.csv\n\n"))
  
  cat(paste("• Total nodes:", vcount(g), "\n"))
  cat(paste("• Total edges:", ecount(g), "\n"))
  cat(paste("• Number of modules:", num_communities, "\n"))
  cat(paste("• Network modularity:", round(modularity(community), 4), "\n\n"))
  
  cat("Module size distribution:\n")
  community_sizes <- sort(table(V(g)$community), decreasing = TRUE)
  for (i in 1:length(community_sizes)) {
    cat(paste("  Module", names(community_sizes)[i], ":", 
              community_sizes[i], "nodes (", 
              round(community_sizes[i]/vcount(g)*100, 1), "%)\n"))
  }
  
  cat("\nNetwork properties:\n")
  cat(paste("  Average degree:", round(mean(degree(g)), 2), "\n"))
  cat(paste("  Network density:", round(edge_density(g), 4), "\n"))
  cat(paste("  Clustering coefficient:", round(transitivity(g, type = "average"), 4), "\n"))
  cat(paste("  Network diameter:", diameter(g), "\n"))
  
  # 计算边颜色统计
  positive_edges <- sum(edge_list$correlation > 0)
  negative_edges <- sum(edge_list$correlation < 0)
  cat("\nEdge color statistics:\n")
  cat(paste("  Red edges (positive correlation):", positive_edges, 
            "(", round(positive_edges/ecount(g)*100, 1), "%)\n"))
  cat(paste("  Green edges (negative correlation):", negative_edges, 
            "(", round(negative_edges/ecount(g)*100, 1), "%)\n"))
  
  cat("\nAnalysis completed successfully!\n")
  cat(rep("=", 60), "\n")
  
} else {
  print("没有找到显著相关的边")
}
# 在现有代码的第21步后面添加：

# 22. 计算ZIPI并生成可视化
cat(rep("=", 60), "\n")
cat("ZIPI ANALYSIS AND VISUALIZATION\n")
cat(rep("=", 60), "\n\n")

# 计算Zi和Pi
comm_members <- V(g)$community
Zi <- rep(0, vcount(g))
Pi <- rep(0, vcount(g))

for (i in 1:vcount(g)) {
  neighbors <- neighbors(g, i)
  
  if (length(neighbors) > 0) {
    within_edges <- 0
    between_edges <- 0
    
    for (neighbor in neighbors) {
      if (comm_members[i] == comm_members[neighbor]) {
        within_edges <- within_edges + 1
      } else {
        between_edges <- between_edges + 1
      }
    }
    
    Zi[i] <- within_edges / length(neighbors)
    Pi[i] <- between_edges / length(neighbors)
  }
}

# 计算ZIPI
ZIPI <- Zi + Pi

# 保存到CSV
zipi_data <- data.frame(
  OTU = V(g)$name,
  Community = V(g)$community,
  Degree = V(g)$degree,
  Zi = Zi,
  Pi = Pi,
  ZIPI = ZIPI
)

write.csv(zipi_data, 
          "D:/5 共现网络分析/1 原始数据/ZIPI_data.csv",
          row.names = FALSE)

# 23. 绘制ZIPI图（Zi vs Pi散点图）- 使用标准阈值
# 保存当前图形参数
old_par <- par(no.readonly = TRUE)

# 设置新的图形参数
par(mar = c(5, 5, 4, 2))

# 根据文献设置标准阈值 (参考：Guimerà & Amaral, 2005)
# Zi阈值：通常使用2.5（标准化后），但我们的Zi在0-1之间，所以用0.5
# Pi阈值：通常使用0.62
zi_threshold <- 0.5    # 对应标准化后的2.5
pi_threshold <- 0.62   # 标准阈值

# 根据标准阈值分类节点类型
node_types <- rep("Peripherals", vcount(g))

# 计算每个节点相对于模块内其他节点的Zi值（标准化）
# 这是更标准的计算方法
calculate_standardized_Zi <- function(g, community) {
  comm_members <- membership(community)
  num_communities <- max(comm_members)
  Zi_standardized <- rep(0, vcount(g))
  
  for (comm in 1:num_communities) {
    nodes_in_comm <- which(comm_members == comm)
    if (length(nodes_in_comm) > 1) {
      # 计算该模块内所有节点的度
      degrees_in_comm <- degree(g)[nodes_in_comm]
      k_mean <- mean(degrees_in_comm)
      k_sd <- sd(degrees_in_comm)
      
      if (k_sd > 0) {
        for (node_idx in nodes_in_comm) {
          k_i <- degree(g)[node_idx]
          # 标准化Zi = (k_i - k_mean) / k_sd
          Zi_standardized[node_idx] <- (k_i - k_mean) / k_sd
        }
      }
    }
  }
  return(Zi_standardized)
}

# 计算标准化的Zi
Zi_standardized <- calculate_standardized_Zi(g, community)

# 重新分类使用标准阈值
# 外围节点：Zi ≤ 2.5 AND Pi ≤ 0.62
# 模块枢纽：Zi > 2.5 AND Pi ≤ 0.62
# 连接器：Zi ≤ 2.5 AND Pi > 0.62
# 网络枢纽：Zi > 2.5 AND Pi > 0.62

node_types <- rep("Peripherals", vcount(g))
node_types[Zi_standardized > 2.5 & Pi <= 0.62] <- "Module hubs"
node_types[Zi_standardized <= 2.5 & Pi > 0.62] <- "Connectors"
node_types[Zi_standardized > 2.5 & Pi > 0.62] <- "Network hubs"

# 设置区域颜色
type_colors <- c(
  "Peripherals" = "gray70",
  "Module hubs" = "#E41A1C",      # 红色
  "Connectors" = "#4DAF4A",       # 绿色
  "Network hubs" = "#377EB8"      # 蓝色
)

# 为每个节点分配颜色
point_colors <- type_colors[node_types]

# 设置点的大小（基于度）
point_size <- sqrt(V(g)$degree + 1) * 0.8

# 创建Zi-Pi散点图（使用标准化Zi）
# 调整x轴范围以适应标准化Zi值
zi_min <- min(Zi_standardized, na.rm = TRUE)
zi_max <- max(Zi_standardized, na.rm = TRUE)

# 扩展边界使图更美观
x_lim <- c(min(zi_min, -3), max(zi_max, 8))
y_lim <- c(0, 1)

plot(Zi_standardized, Pi,
     xlim = x_lim, ylim = y_lim,
     xlab = "Within-module connectivity Z-score (Zi)", 
     ylab = "Among-module connectivity (Pi)",
     main = "Zi-Pi Plot: Network Module Roles with Standard Thresholds",
     pch = 19,  # 实心圆点
     cex = point_size,  # 点大小与度相关
     col = adjustcolor(point_colors, alpha.f = 0.7),
     frame = FALSE,
     cex.lab = 1.2,
     cex.main = 1.3)

# 添加网格
grid(lty = 2, col = "gray85")

# 添加标准阈值线
abline(v = 2.5, lty = 2, col = "red", lwd = 2)      # Zi阈值 = 2.5
abline(h = 0.62, lty = 2, col = "blue", lwd = 2)    # Pi阈值 = 0.62

# 添加阈值标签
text(2.5, 1.05, "Zi = 2.5", col = "red", cex = 0.9, pos = 4)
text(x_lim[1] + 0.5, 0.62, "Pi = 0.62", col = "blue", cex = 0.9, pos = 3)

# 用半透明矩形标记不同区域
# 外围节点区域（左下）
rect(x_lim[1], 0, 2.5, 0.62, 
     col = adjustcolor("gray90", alpha.f = 0.1), border = NA)

# 模块枢纽区域（右下）
rect(2.5, 0, x_lim[2], 0.62, 
     col = adjustcolor("#E41A1C", alpha.f = 0.1), border = NA)

# 连接器区域（左上）
rect(x_lim[1], 0.62, 2.5, y_lim[2], 
     col = adjustcolor("#4DAF4A", alpha.f = 0.1), border = NA)

# 网络枢纽区域（右上）
rect(2.5, 0.62, x_lim[2], y_lim[2], 
     col = adjustcolor("#377EB8", alpha.f = 0.1), border = NA)

# 添加区域标签
text((x_lim[1] + 2.5)/2, 0.31, "Peripherals", 
     cex = 1.0, col = "gray50", font = 2)
text((2.5 + x_lim[2])/2, 0.31, "Module hubs", 
     cex = 1.0, col = "#E41A1C", font = 2)
text((x_lim[1] + 2.5)/2, 0.81, "Connectors", 
     cex = 1.0, col = "#4DAF4A", font = 2)
text((2.5 + x_lim[2])/2, 0.81, "Network hubs", 
     cex = 1.0, col = "#377EB8", font = 2)

# 添加图例
legend("topright",
       legend = c("Peripherals", "Module hubs", "Connectors", "Network hubs"),
       col = c("gray70", "#E41A1C", "#4DAF4A", "#377EB8"),
       pch = 19,
       pt.cex = 1.5,
       bty = "n",
       cex = 0.9,
       title = "Node Types")

# 恢复图形参数
par(old_par)

# 24. 统计各类型数量并打印
type_counts <- table(node_types)
cat("ZIPI Node Type Distribution (Standard Thresholds):\n")
cat(rep("-", 50), "\n")
cat("Thresholds: Zi > 2.5, Pi > 0.62\n")
cat(rep("-", 50), "\n")
for (type in c("Peripherals", "Module hubs", "Connectors", "Network hubs")) {
  if (type %in% names(type_counts)) {
    count <- type_counts[type]
    percentage <- round(count / vcount(g) * 100, 1)
    cat(paste("  ", sprintf("%-15s", type), ": ", 
              sprintf("%3d", count), " nodes (", 
              sprintf("%5.1f", percentage), "%)\n", sep=""))
  }
}

cat("\nZi-Pi Statistics:\n")
cat(rep("-", 40), "\n")
cat(paste("Standardized Zi range: [", 
          round(min(Zi_standardized), 2), ", ", 
          round(max(Zi_standardized), 2), "]\n", sep=""))
cat(paste("Mean standardized Zi:", round(mean(Zi_standardized), 4), "\n"))
cat(paste("Mean Pi:", round(mean(Pi), 4), "\n"))
cat(paste("Nodes with Zi > 2.5:", sum(Zi_standardized > 2.5), 
          " (", round(sum(Zi_standardized > 2.5)/vcount(g)*100, 1), "%)\n", sep=""))
cat(paste("Nodes with Pi > 0.62:", sum(Pi > 0.62), 
          " (", round(sum(Pi > 0.62)/vcount(g)*100, 1), "%)\n", sep=""))
cat(paste("Average ZIPI (Zi+Pi):", round(mean(ZIPI), 4), "\n"))

# 25. 添加标准化Zi到数据框
zipi_data$Zi_standardized <- Zi_standardized

# 按ZIPI排序并显示TOP 10
zipi_sorted <- zipi_data[order(-zipi_data$ZIPI), ]

cat("\nTop 10 OTUs by ZIPI score:\n")
cat(rep("-", 70), "\n")
cat("Note: ZIPI = Zi + Pi (both normalized 0-1)\n")
cat(rep("-", 70), "\n")
for (i in 1:min(10, nrow(zipi_sorted))) {
  otu_name <- zipi_sorted$OTU[i]
  # 截断过长的OTU名称
  if (nchar(otu_name) > 20) {
    otu_name <- paste0(substr(otu_name, 1, 17), "...")
  }
  
  # 确定该OTU的类型
  otu_type <- node_types[which(V(g)$name == zipi_sorted$OTU[i])]
  
  cat(paste(sprintf("%2d", i), ". ", 
            sprintf("%-20s", otu_name),
            " ZIPI:", sprintf("%5.3f", zipi_sorted$ZIPI[i]),
            " Std_Zi:", sprintf("%6.2f", zipi_sorted$Zi_standardized[i]),
            " Pi:", sprintf("%5.3f", zipi_sorted$Pi[i]),
            " Mod:", sprintf("%2d", zipi_sorted$Community[i]),
            " [", otu_type, "]",
            "\n", sep=""))
}

# 26. 保存类型信息到CSV
zipi_data$Node_Type <- node_types
zipi_data <- zipi_data[, c("OTU", "Community", "Degree", 
                           "Zi", "Pi", "ZIPI", "Zi_standardized", "Node_Type")]
write.csv(zipi_data, 
          "D:/5 共现网络分析/1 原始数据/ZIPI_data_with_standard_types.csv",
          row.names = FALSE)

cat("\nFiles saved:\n")
cat("• ZIPI_data.csv - Zi, Pi, and ZIPI scores for all OTUs\n")
cat("• ZIPI_data_with_standard_types.csv - With standardized Zi and node classification\n")
cat("• Zi-Pi plot displayed in graphics window (with standard thresholds)\n")
cat(rep("=", 60), "\n")

# 27. 可选：保存高分辨率图片
png("D:/5 共现网络分析/1 原始数据/Zi_Pi_plot_standard.png", 
    width = 3200, height = 2800, res = 300)

par(mar = c(5, 5, 4, 2))
plot(Zi_standardized, Pi,
     xlim = x_lim, ylim = y_lim,
     xlab = "Within-module connectivity Z-score (Zi)", 
     ylab = "Among-module connectivity (Pi)",
     main = "Zi-Pi Plot: Network Module Roles with Standard Thresholds",
     pch = 19,
     cex = point_size,
     col = adjustcolor(point_colors, alpha.f = 0.7),
     frame = FALSE,
     cex.lab = 1.2,
     cex.main = 1.3)

grid(lty = 2, col = "gray85")
abline(v = 2.5, lty = 2, col = "red", lwd = 2)
abline(h = 0.62, lty = 2, col = "blue", lwd = 2)

text(2.5, 1.05, "Zi = 2.5", col = "red", cex = 0.9, pos = 4)
text(x_lim[1] + 0.5, 0.62, "Pi = 0.62", col = "blue", cex = 0.9, pos = 3)

# 添加区域标签
text((x_lim[1] + 2.5)/2, 0.31, "Peripherals", cex = 1.0, col = "gray50", font = 2)
text((2.5 + x_lim[2])/2, 0.31, "Module hubs", cex = 1.0, col = "#E41A1C", font = 2)
text((x_lim[1] + 2.5)/2, 0.81, "Connectors", cex = 1.0, col = "#4DAF4A", font = 2)
text((2.5 + x_lim[2])/2, 0.81, "Network hubs", cex = 1.0, col = "#377EB8", font = 2)

legend("topright",
       legend = c("Peripherals", "Module hubs", "Connectors", "Network hubs"),
       col = c("gray70", "#E41A1C", "#4DAF4A", "#377EB8"),
       pch = 19,
       pt.cex = 1.5,
       bty = "n",
       cex = 0.9,
       title = "Node Types")

dev.off()

cat("• High-resolution image saved: Zi_Pi_plot_standard.png\n")
cat(rep("=", 60), "\n")

# 28. 关键发现总结
cat("\nKEY FINDINGS:\n")
cat(rep("-", 60), "\n")
if (sum(node_types == "Network hubs") > 0) {
  cat("• Found", sum(node_types == "Network hubs"), "network hub(s) - critical for network stability\n")
}
if (sum(node_types == "Module hubs") > 0) {
  cat("• Found", sum(node_types == "Module hubs"), "module hub(s) - important within modules\n")
}
if (sum(node_types == "Connectors") > 0) {
  cat("• Found", sum(node_types == "Connectors"), "connector(s) - link different modules\n")
}
cat("• Majority of nodes (", sum(node_types == "Peripherals"), ") are peripherals\n", sep="")
cat(rep("=", 60), "\n")