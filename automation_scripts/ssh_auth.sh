cat authorized_keys > id_rsa.pub
cat diml_old_private_no_passphrase_mac.pem > id_rsa
chmod 400 ~/.ssh/id_rsa
chmod 400 ~/.ssh/id_rsa.pub
ssh-copy-id diml_2022@big_instance_1
ssh-copy-id diml_2022@big_instance_2
ssh-copy-id diml_2022@big_instance_3
ssh-copy-id diml_2022@big_instance_4
ssh-copy-id diml_2022@big_instance_5
ssh-copy-id diml_2022@big_instance_6
ssh-copy-id diml_2022@big_instance_7
ssh-copy-id diml_2022@big_instance_8