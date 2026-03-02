# Mintlify Starter Deployment Plan

This plan uses the Mintlify Starter repository as the base and deploys a long‑running Mintlify dev server behind Nginx. It also includes domain acquisition using DigitalPlat FreeDomain.

## 1. Project Setup (Mintlify Starter)

Option A: Use Mintlify Starter repository

```bash
mkdir -p ~/handover-docs
cd ~/handover-docs
git clone https://github.com/mintlify/starter.git .
npm install
```

Option B: Use Mintlify init

```bash
mkdir -p ~/handover-docs
cd ~/handover-docs
npx mintlify init .
npm install
```

Run locally to verify:

```bash
npx mintlify dev
```

## 2. Domain (DigitalPlat FreeDomain)

DigitalPlat provides free domains such as:

- .dpdns.org
- .us.kg
- .qzz.io
- .xx.kg

Recommended flow:

1. Register a domain in the DigitalPlat FreeDomain dashboard.
2. Use a DNS provider (Cloudflare, FreeDNS, or Hostry).
3. Create A records to point the domain to your server public IP.
4. Set Nginx server_name to the chosen domain.

## 3. Long‑Running Service (systemd)

Create a systemd service to keep Mintlify running and restart on failure.

```bash
sudo nano /etc/systemd/system/mintlify.service
```

```ini
[Unit]
Description=Mintlify Dev Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/你的用户名/handover-docs
ExecStart=/usr/bin/npx mintlify dev
Restart=always
RestartSec=5
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mintlify
sudo systemctl start mintlify
sudo systemctl status mintlify
```

## 4. Nginx Reverse Proxy

```bash
sudo apt-get update
sudo apt-get install nginx
sudo nano /etc/nginx/sites-available/handover-docs
```

```nginx
server {
    listen 80;
    server_name example.dpdns.org;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
        proxy_connect_timeout 5s;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/handover-docs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 5. HTTPS (Optional, Recommended)

Use Certbot for TLS:

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d example.dpdns.org
```

## 6. Operations

Check status and logs:

```bash
systemctl status mintlify
journalctl -u mintlify -f
systemctl status nginx
```

