# usage: mkconst.sh name stir light harv
cat > $1.py <<EOT
STIR = $2
LIGHT = $3
HARV = $4
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (STIR, LIGHT, HARV)
EOT
