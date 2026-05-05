while read -r N D r; do
    julia track_data_statistics.jl $N $D $r
done <<EOF
2 3 4
2 3 5
2 3 6
2 4 5
2 4 6
2 4 7
2 5 6
2 5 7
2 5 8
3 3 4
3 3 5
3 3 6
3 4 5
3 4 6
3 4 7
3 5 6
3 5 7
3 5 8
EOF
