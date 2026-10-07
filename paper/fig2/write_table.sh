while read -r N D r; do
    julia track_data_statistics.jl $N $D $r
done <<EOF
2 3 3
2 3 4
2 3 5
2 4 3
2 4 4
2 4 5
2 5 3
2 5 4
2 5 5
3 3 3
3 3 4
3 3 5
3 4 3
3 4 4
3 4 5
3 5 3
3 5 4
3 5 5
EOF
