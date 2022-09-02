#this file uses the package minimal generator, publicly available here: https://github.com/TDAMinimalGeneratorResearch/minimal-generator
# it has to be compiled in the same folder, after compiling the files installRequirements.jl and exampleRun.jl

## 1. We load the file from Dropbox
pc_CD = readdlm("/Users/jeannefernandez/Dropbox/Optimal_representatives_weather/data/Charney_Devore/charney_devore.txt") #todo: adapt the path
pc_CD = transpose(pc_CD)

#we plot the 3 first coordinates
Plots.scatter(pc_CD[1,:],pc_CD[2,:], pc_CD[3,:])

#we take a random subset to be able to do local computations
subsample_size = 100
S_CD = sample(1:size(pc_CD)[2],subsample_size,replace=false);
pc_CD_sub = view(pc_CD, :, S_CD);
Plots.scatter(pc_CD_sub[1,:],pc_CD_sub[2,:], pc_CD_sub[3,:])

#checking the dimension and the number of points in the datset:
print("The dataset is in ", size(pc_CD_sub)[1], "D, and has ", size(pc_CD_sub)[2], " points.")


## 2. We call computeHomology

C = computeHomology(pc_CD_sub, false, 1) # <- compute homology of the pointcloud in dimension 1

plotBarCode(C) # plots the bar code for a pointcloud
PlotlyJS.savefig(plotBarCode(C), "Images/barcode_charney_devore.png") #we see that the longest feature in dim 1 is for l=11


## 3. Minimal generators from exampleRun.jl

##########################################
#              MINIMAL GENERATOR         #
##########################################


Pkg.build("Gurobi") # run in repl: ENV["GUROBI_HOME"]="/Library/gurobi903/mac64/"
using Gurobi
d = 1  # dimension of the generator we hope to minimize
requireIntegralSol = false # whether we want to require the generator vector to have integral entries.

## 4. PCA(3)

M = MultivariateStats.fit(PCA, pc_CD_sub; maxoutdim=3)
pc_pca = MultivariateStats.transform(M, pc_CD_sub)
#print(pc_a)
display(M)
C_pca = computeHomology(pc_pca, false, 1) # <- compute homology of pc in dimension 1
plotBarCode(C_pca) # plots the bar code for a pointcloud
PlotlyJS.savefig(plotBarCode(C_pca), "Images/barcode_charney_devore_pca.png") #we see that the longest feature in dim 1 is for l=12


##########################################
#              EDGE- LOSS                #
##########################################


##################
# We optimize a single generator, to be chosen in the barcode as the index of the longuest bar
###################

l = 11; # todo: adapt the index of the generator you wish to minimize, usually the index of the longuest bar in the barcode
d=1
len_weighted = false # can substitute with a vector of lengths to customize your own way of defining length of edges
uniform_weighted_minimal_gen, uniform_Len, uniform_zeroNorm = C_d_l__minimal(C,d,l, len_weighted, false, C.generators[1][1:l-1], false)

len_weighted = true
length_weighted_minimal_gen, len_Len, len_zeroNorm = C_d_l__minimal(C,d,l, true, len_weighted, C.generators[1][1:l-1], false)

# plot only works up to dimension 3.
plt_unif_min = plotMinimalEdgeGenerators(C,1,uniform_weighted_minimal_gen) # plots the optimal generator
plt_length_min = plotMinimalEdgeGenerators(C,1,length_weighted_minimal_gen) # plots the optimal generator
plt_gen = plotGenerators(C,d,l) # plots the original generator

PlotlyJS.savefig(plt_unif_min, "Images/CD_unif_min.html")
PlotlyJS.savefig(plt_length_min, "Images/CD_length_min.html")
PlotlyJS.savefig(plt_gen, "Images/CD_gen.html")


##################
# On PCA embedding
###################

l = 12; # index of the generator we hope to minimize
d=1
len_weighted = false # can substitute with a vector of lengths to customize your own way of defining length of edges
uniform_weighted_minimal_gen, uniform_Len, uniform_zeroNorm = C_d_l__minimal(C_pca,d,l, len_weighted, false, C.generators[1][1:l-1], false)

len_weighted = true
length_weighted_minimal_gen, len_Len, len_zeroNorm = C_d_l__minimal(C_pca,d,l, true, len_weighted, C.generators[1][1:l-1], false)

# plot only works up to dimension 3.
plt_unif_min = plotMinimalEdgeGenerators(C_pca,1,uniform_weighted_minimal_gen) # plots the optimal generator
plt_length_min = plotMinimalEdgeGenerators(C_pca,1,length_weighted_minimal_gen) # plots the optimal generator
plt_gen = plotGenerators(C_pca,d,l) # plots the original generator

PlotlyJS.savefig(plt_unif_min, "Images/CD_unif_min_pca.html")
PlotlyJS.savefig(plt_length_min, "Images/CD_length_min_pca.html")
PlotlyJS.savefig(plt_gen, "Images/CD_gen_pca.html")


##########################################
#         TRIANGLE - LOSS                #
##########################################

# Optimize a single generator
dd, triVerts = constructInput(C,d,l)
optimal_cycle = findAreaOptimalCycle(C, d, l, dd, triVerts, true)[1]
optimal_volume = findAreaOptimalCycle(C, d, l, dd, triVerts, true)[2] # number of triangles this cycle bounds

plt_optimal_cycle = plotMinimalEdgeGenerators(C,1,optimal_cycle) # plots the optimal generator
plt_optimal_volume = plotMinimalEdgeGenerators(C,1,optimal_volume) # plots the optimal generator

PlotlyJS.savefig(plt_unif_min, "Images/CD_min_unif_triangle.html")
PlotlyJS.savefig(plt_length_min, "Images/CD_min_length_triangle.html")



##################
# On PCA embedding
###################

# Optimize a single generator
dd, triVerts = constructInput(C_pca,d,l)
optimal_cycle = findAreaOptimalCycle(C_pca, d, l, dd, triVerts, true)[1]
optimal_volume = findAreaOptimalCycle(C_pca, d, l, dd, triVerts, true)[2] # number of triangles this cycle bounds

plt_optimal_cycle = plotMinimalEdgeGenerators(C_pca,1,optimal_cycle) # plots the optimal generator
plt_optimal_volume = plotMinimalEdgeGenerators(C_pca,1,optimal_volume) # plots the optimal generator

PlotlyJS.savefig(plt_unif_min, "Images/CD_min_unif_triangle_pca.html")
PlotlyJS.savefig(plt_length_min, "Images/CD_min_length_triangle_pca.html")
