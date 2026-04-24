# python scripts/train.py \
# 	data=lp_pdbbind \
# 	model=mlp_baseline \
# 	run_name=lp_mlp_base_v2 \
# 	device=cuda \
# 	training.max_epochs=100 \
# 	training.lr=2e-4 \
# 	training.weight_decay=1e-5 \
# 	model.dropout=0.1 \
# 	training.batch_size=64 \
# 	training.patience=20 \
# 	"model.hidden_dims=[128,64]" \
# 	training.loss=huber

#  python scripts/train.py \
#  	data=lp_pdbbind \
# 	model=egnn \
# 	run_name=lp_egnn_v1 \
# 	device=cuda \
# 	training.max_epochs=300 \
# 	training.lr=1.5e-4 \
# 	training.weight_decay=1e-4 \
# 	model.dropout=0.1 \
# 	training.batch_size=16 \
# 	training.patience=30 \
# 	training.loss=combined \
# 	training.combined_pearson_weight=0.5 \
# 	training.scheduler.warmup_epochs=15 

python scripts/train.py \
	data=lp_pdbbind \
	data.processed_dir=data/processed/lp_pdbbind_geom \
	model=egnn \
	model.edge_attr_dim=6 \
	run_name=lp_egnn_geom_v1 \
	device=cuda \
	training.max_epochs=300 \
	training.lr=1.5e-4 \
	training.weight_decay=1e-4 \
	model.dropout=0.1 \
	training.batch_size=16 \
	training.patience=30 \
	training.loss=combined \
	training.combined_pearson_weight=0.5 \
	training.scheduler.warmup_epochs=15