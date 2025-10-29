# Development
mix phx.server

# Production
MIX_ENV=prod mix release
_build/prod/rel/alphafold3_gateway/bin/alphafold3_gateway start
