function sigmoid(x)
	inv(1 + exp(-x))
end;

function logit(ϕ)
	log(ϕ / (1 - ϕ))
end;
